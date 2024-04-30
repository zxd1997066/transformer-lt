#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import torch
import os
import time
from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    if args.precision == "bfloat16":
        # with torch.amp.autocast(enabled=True, configure=torch.bfloat16, torch.no_grad(): 
        print("Running with bfloat16...")

    use_cuda = torch.cuda.is_available()

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()
        if args.channels_last:
            model_oob = model
            try:
                model_oob = model_oob.to(memory_format=torch.channels_last)
                print("---- Use channels last format.")
            except:
                print("---- Use normal format.")
            model = model_oob
        if args.compile:
            model = torch.compile(model, backend=args.backend, options={"freezing": True})
        if args.ipex:
            import intel_extension_for_pytorch as ipex
            if args.precision == 'bfloat16':
                model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
                print('Running with bfloat16...')
            else:
                model = ipex.optimize(model, dtype=torch.float32, inplace=True)
                print('Running with float32...')
        if args.jit:
            try:
                model = torch.jit.script(model)
                print("---- With JIT enabled.")
            except:
                print("---- With JIT disabled.")
            if args.ipex:
                model = torch.jit.freeze(model)
        if args.compile:
            model = torch.compile(model, backend=args.backend, options={"freezing": True})

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)
    if args.compile:
        model = torch.compile(model, backend=args.backend, options={"freezing": True})
    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    batch_time_list = []
    has_target = True
    with torch.no_grad():
        with progress_bar.build_progress_bar(args, itr) as t:
            wps_meter = TimeMeter()
            for iters_runned, sample in enumerate(t):
                if args.max_iters > 0 and iters_runned >= args.max_iters + args.warmup_iters:
                    break
                if use_cuda:
                    sample = utils.move_to_cuda(sample) if use_cuda else sample
                if 'net_input' not in sample:
                    continue

                prefix_tokens = None
                if args.prefix_size > 0:
                    prefix_tokens = sample['target'][:, :args.prefix_size]

                tic = time.time()
                if iters_runned >= args.warmup_iters:
                    gen_timer.start()
                if args.profile:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
                        hypos = task.inference_step(generator, models, sample, prefix_tokens)
                    if  iters_runned == int((args.max_iters + args.warmup_iters)/2):
                        import pathlib
                        timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                        if not os.path.exists(timeline_dir):
                            os.makedirs(timeline_dir)
                        timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                                    "transformerlt" + str(iters_runned) + '-' + str(os.getpid()) + '.json'
                        print(timeline_file)
                        prof.export_chrome_trace(timeline_file)
                        table_res = prof.key_averages().table(sort_by="cpu_time_total")
                        print(table_res)
                        # self.save_profile_result(timeline_dir + torch.backends.quantized.engine + "_result_average.xlsx", table_res)
                else:
                    hypos = task.inference_step(generator, models, sample, prefix_tokens)
                num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
                if iters_runned >= args.warmup_iters:
                    gen_timer.stop(num_generated_tokens)
                toc = time.time()
                print("Iteration: {}, inference time: {} sec.".format(iters_runned, toc - tic), flush=True)
                if iters_runned >= args.warmup_iters:
                    batch_time_list.append((toc - tic) * 1000)
                if iters_runned >= args.warmup_iters:
                    num_sentences += sample['nsentences']
                continue
                for i, sample_id in enumerate(sample['id'].tolist()):
                    has_target = sample['target'] is not None

                    # Remove padding
                    src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                    target_tokens = None
                    if has_target:
                        target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

                    # Either retrieve the original sentences or regenerate them from tokens.
                    if align_dict is not None:
                        src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                        target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                    else:
                        if src_dict is not None:
                            src_str = src_dict.string(src_tokens, args.remove_bpe)
                        else:
                            src_str = ""
                        if has_target:
                            target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                    if not args.quiet:
                        if src_dict is not None:
                            print('S-{}\t{}'.format(sample_id, src_str.encode("utf-8")))
                        if has_target:
                            print('T-{}\t{}'.format(sample_id, target_str.encode("utf-8")))

                    # Process top predictions
                    for j, hypo in enumerate(hypos[i][:args.nbest]):
                        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                            hypo_tokens=hypo['tokens'].int().cpu(),
                            src_str=src_str,
                            alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                            align_dict=align_dict,
                            tgt_dict=tgt_dict,
                            remove_bpe=args.remove_bpe,
                        )

                        if not args.quiet:
                            print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str.encode("utf-8")))
                            print('P-{}\t{}'.format(
                                sample_id,
                                ' '.join(map(
                                    lambda x: '{:.4f}'.format(x),
                                    hypo['positional_scores'].tolist(),
                                ))
                            ))

                            if args.print_alignment:
                                print('A-{}\t{}'.format(
                                    sample_id,
                                    ' '.join(map(lambda x: str(utils.item(x)), alignment))
                                ))

                        # Score only the top hypothesis
                        if has_target and j == 0:
                            if align_dict is not None or args.remove_bpe is not None:
                                # Convert back to tokens for evaluation with unk replacement and/or without BPE
                                target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                            if hasattr(scorer, 'add_string'):
                                scorer.add_string(target_str, hypo_str)
                            else:
                                scorer.add(target_tokens, hypo_tokens)

                wps_meter.update(num_generated_tokens)
                t.log({'wps': round(wps_meter.avg)})

    # print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
    #     num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    # if has_target:
    #     print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))

    print("\n", "-"*20, "Summary", "-"*20)
    latency = gen_timer.sum / num_sentences * 1000
    throughput = num_sentences / gen_timer.sum
    print("inference latency:\t {:.3f} ms".format(latency))
    print("inference Throughput:\t {:.2f} samples/s".format(throughput))
    # P50
    batch_time_list.sort()
    p50_latency = batch_time_list[int(len(batch_time_list) * 0.50) - 1]
    p90_latency = batch_time_list[int(len(batch_time_list) * 0.90) - 1]
    p99_latency = batch_time_list[int(len(batch_time_list) * 0.99) - 1]
    print('Latency P50:\t %.3f ms\nLatency P90:\t %.3f ms\nLatency P99:\t %.3f ms\n'\
            % (p50_latency, p90_latency, p99_latency))


    return scorer


def save_profile_result(filename, table):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    keys = ["Name", "Self CPU total %", "Self CPU total", "CPU total %" , "CPU total", \
            "CPU time avg", "Number of Calls"]
    for j in range(len(keys)):
        worksheet.write(0, j, keys[j])

    lines = table.split("\n")
    for i in range(3, len(lines)-4):
        words = lines[i].split(" ")
        j = 0
        for word in words:
            if not word == "":
                worksheet.write(i-2, j, word)
                j += 1
    workbook.close()



def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    if args.precision == "bfloat16":
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.bfloat16):
            main(args)
    elif args.precision == "float16":
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.half):
            main(args)
    else:
        main(args)


if __name__ == '__main__':
    cli_main()
