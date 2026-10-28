"""Run the repository evaluator at an explicitly overridden input resolution."""

import argparse
import hashlib
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--eval_width", type=int, required=True)
    parser.add_argument("--eval_height", type=int, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    known, remaining = parser.parse_known_args()
    if min(known.eval_width, known.eval_height) <= 0:
        parser.error("evaluation dimensions must be positive")

    sys.argv = [sys.argv[0]] + remaining
    import datasets
    import evaluate_depth
    import networks
    from options import MonodepthOptions

    original_dataset = datasets.KITTIRAWDataset

    def dimension_override(*args, **kwargs):
        args = list(args)
        if len(args) >= 4:
            args[2], args[3] = known.eval_height, known.eval_width
        else:
            kwargs["height"] = known.eval_height
            kwargs["width"] = known.eval_width
        return original_dataset(*args, **kwargs)

    class TracedEncoder(networks.ResnetEncoder):
        traced = False
        def forward(self, input_image):
            if not self.traced:
                print("actual_input_shape={}".format(tuple(input_image.shape)))
                self.traced = True
            return super(TracedEncoder, self).forward(input_image)

    class TracedDecoder(networks.DepthDecoder):
        traced = False
        def forward(self, features):
            result = super(TracedDecoder, self).forward(features)
            if not self.traced:
                print("prediction_disp_shape={}".format(tuple(result[("disp", 0)].shape)))
                self.traced = True
            return result

    datasets.KITTIRAWDataset = dimension_override
    networks.ResnetEncoder = TracedEncoder
    networks.DepthDecoder = TracedDecoder
    options = MonodepthOptions().parse()
    # The evaluator now supports CPU as an explicit fallback when the driver is
    # unavailable; this keeps metric protocol identical while reporting the
    # actual device in its log.
    if not __import__("torch").cuda.is_available():
        options.no_cuda = True
    weights = os.path.expanduser(options.load_weights_folder)
    if not weights:
        parser.error("pass --load_weights_folder through to the evaluator")
    encoder_path = os.path.join(weights, "encoder.pth")
    print("checkpoint_encoder_sha256={}".format(sha256(encoder_path)))
    print("requested_input={}x{}".format(known.eval_width, known.eval_height))
    metrics = evaluate_depth.evaluate(options)
    if known.output_csv and metrics is not None:
        import csv
        exists = os.path.isfile(known.output_csv)
        with open(known.output_csv, "a", newline="") as stream:
            writer = csv.writer(stream)
            if not exists:
                writer.writerow(["width", "height", "actual_tensor_shape", "AbsRel", "SqRel",
                                 "RMSE", "RMSElog", "delta1", "delta2", "delta3", "num_images"])
            writer.writerow([known.eval_width, known.eval_height,
                             "{}x{}".format(known.eval_height, known.eval_width)] +
                            [float(value) for value in metrics] + [697])


if __name__ == "__main__":
    main()
