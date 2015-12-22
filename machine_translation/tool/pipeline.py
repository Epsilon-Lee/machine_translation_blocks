#!/usr/bin/env python
import argparse
import cPickle
import gzip
import sys
import uuid
import numpy
import os
import subprocess
import logging
from picklable_itertools.extras import equizip

parser = argparse.ArgumentParser()
parser.add_argument("--source_input",
                    type=str,
                    help="The source input file name")
parser.add_argument("--target_input",
                    type=str,
                    help="The target input file name")
parser.add_argument("--source",
                    type=str,
                    help="The source language")
parser.add_argument("--target",
                    type=str,
                    help="The target language")
parser.add_argument("--source_vocab",
                    type=str,
                    help="The source language")
parser.add_argument("--target_vocab",
                    type=str,
                    help="The target language")
parser.add_argument("--dev_source",
                    type=str,
                    help="The source language")
parser.add_argument("--dev_target",
                    type=str,
                    help="The target language")

def tokenize_text_files(files_to_tokenize, tokenizer):
    for name in files_to_tokenize:
        logger.info("Tokenizing file [{}]".format(name))
        out_file = os.path.join(
            OUTPUT_DIR, os.path.basename(name) + '.tok')
        logger.info("...writing tokenized file [{}]".format(out_file))
        var = ["perl", tokenizer,  "-l", name.split('.')[-1]]
        if not os.path.exists(out_file):
            with open(name, 'r') as inp:
                with open(out_file, 'w', 0) as out:
                    subprocess.check_call(
                        var, stdin=inp, stdout=out, shell=False)
        else:
            logger.info("...file exists [{}]".format(out_file))

def create_vocabularies(tr_files, preprocess_file):
    src_vocab_name = os.path.join(
        OUTPUT_DIR, 'vocab.{}-{}.{}.pkl'.format(
            args.source, args.target, args.source))
    trg_vocab_name = os.path.join(
        OUTPUT_DIR, 'vocab.{}-{}.{}.pkl'.format(
            args.source, args.target, args.target))
    src_filename = os.path.basename(
        tr_files[[i for i, n in enumerate(tr_files)
                  if n.endswith(args.source)][0]]) + '.tok'
    trg_filename = os.path.basename(
        tr_files[[i for i, n in enumerate(tr_files)
                  if n.endswith(args.target)][0]]) + '.tok'
    logger.info("Creating source vocabulary [{}]".format(src_vocab_name))
    if not os.path.exists(src_vocab_name):
        subprocess.check_call(" python {} -d {} -v {} {}".format(
            preprocess_file, src_vocab_name, args.source_vocab,
            os.path.join(OUTPUT_DIR, src_filename)),
            shell=True)
    else:
        logger.info("...file exists [{}]".format(src_vocab_name))

    logger.info("Creating target vocabulary [{}]".format(trg_vocab_name))
    if not os.path.exists(trg_vocab_name):
        subprocess.check_call(" python {} -d {} -v {} {}".format(
            preprocess_file, trg_vocab_name, args.target_vocab,
            os.path.join(OUTPUT_DIR, trg_filename)),
            shell=True)
    else:
        logger.info("...file exists [{}]".format(trg_vocab_name))
    return src_filename, trg_filename

def merge_parallel(src_filename, trg_filename, merged_filename):
    with open(src_filename, 'r') as left:
        with open(trg_filename, 'r') as right:
            with open(merged_filename, 'w') as final:
                for lline, rline in equizip(left, right):
                    if (lline != '\n') and (rline != '\n'):
                        final.write(lline[:-1] + ' ||| ' + rline)


def split_parallel(merged_filename, src_filename, trg_filename):
    with open(merged_filename) as combined:
        with open(src_filename, 'w') as left:
            with open(trg_filename, 'w') as right:
                for line in combined:
                    line = line.split('|||')
                    left.write(line[0].strip() + '\n')
                    right.write(line[1].strip() + '\n')


def shuffle_parallel(src_filename, trg_filename):
    logger.info("Shuffling jointly [{}] and [{}]".format(src_filename,trg_filename))
    out_src = src_filename + '.shuf'
    out_trg = trg_filename + '.shuf'
    merged_filename = str(uuid.uuid4())
    shuffled_filename = str(uuid.uuid4())
    if not os.path.exists(out_src) or not os.path.exists(out_trg):
        try:
            merge_parallel(src_filename, trg_filename, merged_filename)
            subprocess.check_call(
                " shuf {} > {} ".format(merged_filename, shuffled_filename),
                shell=True)
            split_parallel(shuffled_filename, out_src, out_trg)
            logger.info(
                "...files shuffled [{}] and [{}]".format(out_src, out_trg))
        except Exception as e:
            logger.error("{}".format(str(e)))
    else:
        logger.info("...files exist [{}] and [{}]".format(out_src, out_trg))
    if os.path.exists(merged_filename):
        os.remove(merged_filename)
    if os.path.exists(shuffled_filename):
        os.remove(shuffled_filename)

if __name__ == "__main__":
    OUTPUT_DIR='./'
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = parser.parse_args()
    tokenize_text_files([args.source_input,args.target_input],'../tool/tokenizer.perl')
    tokenize_text_files([args.dev_source,args.dev_target],'../tool/tokenizer.perl')
    src_filename, trg_filename = create_vocabularies([args.source_input,args.target_input], '../tool/preprocess.py')
    shuffle_parallel(src_filename,trg_filename)
