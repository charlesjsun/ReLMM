import argparse
import json

def process_params(params):
    return json.loads(params.replace("\\quote", "\""))

def main(args):
    print('experiment ran with')
    print(args.params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--params",type=process_params, default='{}')
    args = parser.parse_args()

    main(args)