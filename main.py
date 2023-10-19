from glimmer import GLIMMER
import argparse
from utils import read_file


def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="ttr", choices=['ttr', 'distance', 'eigengap'])
    parser.add_argument("--input_file_path", type=str, default="dataset/multi-news/test.truncate.fix.pun.src.txt")
    parser.add_argument("--output_file_path", type=str, default="Summary.txt",
                        help="please ensure that the file does not exist or is empty")

    return parser.parse_args()


def run_glimmer():
    args = read_arguments()
    input_file_path = args.input_file_path
    output_file_path = args.output_file_path
    method = args.method

    src_list = read_file(input_file_path)
    print(len(src_list), "samples in total")

    glimmer = GLIMMER(output_file_path=output_file_path, method=method)
    summaries = glimmer.summarize(src_list)


if __name__ == "__main__":
    run_glimmer()
