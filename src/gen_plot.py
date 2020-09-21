import sys
import os
from collections import defaultdict

import numpy as np

from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record


def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def aggregate(path):
    keys = [
        'Eval/Reward',
        # 'Eval/Length',
        # 'Eval/MCTS_Confidence',
        'Train/AvgLoss'
    ]

    agg_runs = defaultdict(lambda: defaultdict(list))
    agg_keys = {}

    path = os.path.join(path, "event.tfevents")
    for event in my_summary_iterator(path):
        tag = event.summary.value[0].tag
        for key in keys:
            if tag.startswith(key):
                run = tag[tag.rfind('/')+1:]
                agg_runs[key][run].append(event.summary.value[0].simple_value)

    for key in agg_runs:
        aggregated = []
        for run in agg_runs[key]:
            aggregated.append(agg_runs[key][run])
        max_len = max(len(x) for x in aggregated)
        aggregated_ = []
        for x in aggregated:
            aggregated_.append(
                np.pad(
                    x,
                    (0, max_len - len(x)),
                    mode='constant',
                    constant_values=(0, x[-1])
                )
            )
        agg_keys[key] = np.array(aggregated_)

    shape_before = agg_keys[next(iter(agg_runs.keys()))].shape
    data_ = agg_keys[next(iter(agg_runs.keys()))]
    # # TODO !!! This is very specific. However, since I know my experiments..
    # It means the last run was not finished yet. So we cannot use it.
    if data_[-1][-1] < 195 and data_[-1].shape[0] < 1500:
        for key in agg_runs:
            agg_keys[key] = agg_keys[key][:-1]
    print('filter', shape_before, agg_keys[next(iter(agg_runs.keys()))].shape)
    return agg_runs, agg_keys


def gen_tex_single_key(data, key):
    new_data = [
        smooth(np.percentile(data, percentile, axis=0), 20)[10:-9]
        for percentile in [50, 10, 25, 75, 90]
    ]
    coords = [
        "".join(str((i, round(x, 3))) for i, x in enumerate(nd))
        for nd in new_data
    ]

    red_line = ""
    if "Eval" in key:
        red_line = "".join(["(%d, 195)" % i for i, _ in enumerate(data[0])])

    return """
        \\begin{tikzpicture}
            \\begin{axis}[
                thick,smooth,no markers,
                grid=both,
                grid style={line width=.1pt, draw=gray!10},
                xlabel={Steps},
                ylabel={%s}]
            ]
            \\addplot+[name path=A,black,line width=1pt] coordinates {%s};
            \\addplot+[name path=B,black,line width=.1pt] coordinates {%s};
            \\addplot+[name path=C,black,line width=.1pt] coordinates {%s};
            \\addplot+[name path=D,black,line width=.1pt] coordinates {%s};
            \\addplot+[name path=E,black,line width=.1pt] coordinates {%s};

            \\addplot+[name path=ZZZ,red,line width=.3pt] coordinates {%s};

            \\addplot[blue!50,fill opacity=0.2] fill between[of=A and B];
            \\addplot[blue!50,fill opacity=0.2] fill between[of=A and C];
            \\addplot[blue!50,fill opacity=0.2] fill between[of=A and D];
            \\addplot[blue!50,fill opacity=0.2] fill between[of=A and E];
            \\end{axis}
        \\end{tikzpicture}
        """ % (
            key, *coords, red_line
        )


def gen_tex_single(agg_runs, agg_keys):
    gen = """\\documentclass{standalone}
        \\usepackage{tikz,pgfplots}

        \\pgfplotsset{compat=1.10}

        \\usepgfplotslibrary{fillbetween,external}
        \\tikzexternalize

        \\begin{document}
        """
    for key in agg_runs:
        data = agg_keys[key]
        gen += gen_tex_single_key(data, key)
    gen += "\\end{document}"
    return gen


def gen_tex_multiple(agg_runs, agg_keys):
    gen = """\\documentclass{standalone}
        \\usepackage{tikz,pgfplots}

        \\pgfplotsset{compat=1.10}

        \\usepgfplotslibrary{fillbetween,external}
        \\tikzexternalize

        \\begin{document}
        """
    data = {}
    for key in agg_runs:
        data = agg_keys[key]
        gen += gen_tex_single(data, key)
    gen += "\\end{document}"
    return gen


def run():
    if len(sys.argv) < 2:
        for filename in os.listdir('runs/'):
            path = os.path.join('runs/', filename)
            if not os.path.isdir(path):
                continue
            print('Example path: ', path)
            return

    if len(sys.argv) == 2:
        path = sys.argv[1]
        print('Loading', path)
        agg_runs, agg_keys = aggregate(path)
        gen = gen_tex_single(agg_runs, agg_keys)
        with open('testingtex/a.tex', 'w+') as f:
            f.write(gen)
        # In case I want to automate pdflatex too.
        # os.system('cd testingtex && pdflatex --shell-escape a.tex && cd ..')

    if len(sys.argv) > 2:
        data = {}
        for path in sys.argv[1:]:
            print('Loading', path)
            agg_runs, agg_keys = aggregate(path)
            for key in agg_runs:
                if key not in data:
                    data[key] = agg_keys[key]
                else:
                    xlen = max(data[key].shape[1], agg_keys[key].shape[1])

                    X = data[key]
                    Y = agg_keys[key]
                    if xlen > X.shape[1]:
                        X = np.pad(X, ((0, 0), (0, xlen - X.shape[1])), mode='edge')
                    if xlen > Y.shape[1]:
                        Y = np.pad(Y, ((0, 0), (0, xlen - Y.shape[1])), mode='edge')

                    # print(data[key].shape, agg_keys[key].shape)
                    # print(X.shape, Y.shape)
                    data[key] = np.concatenate((X, Y))
        print('total', data[next(iter(agg_runs.keys()))].shape)
        gen = gen_tex_single(agg_runs, data)
        with open('testingtex/a.tex', 'w+') as f:
            f.write(gen)


if __name__ == '__main__':
    run()
