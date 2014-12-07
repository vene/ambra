"""
Readers for the SemEval 2014 task 7 dataset distribution
========================================================

"""

# Author: Vlad Niculae <vn66@cornell.edu>
# License: Simplified BSD

from __future__ import print_function
import re


def _interval_from_str(s):
    """Turn e.g. "1981-1988" into [1981, 1988]"""
    return map(int, s.split("-"))


def read_semeval_data(f_content, granularity='fine'):
    """Turn the SemEval pseudo-xml into python-friendly JSON.

    Parameters
    ----------

    f_content: unicode,
        The contents of the file to be parsed.

    granularity: {'fine'|'medium'|'coarse'}, default: 'fine'
        The granularity to operate on.

    Returns
    -------

    texts: list of dicts,
        List of dict records with `id`, `true_interval`,
        `possible_intervals` and `text` for each entry in the file.
    """

    # we have to use regex to extract this... it's the cleanest way, I think,
    # because the xml is malformed
    text_re = re.compile("<text.*?</text>", re.DOTALL)
    id_re = re.compile('id="(.*?)"')
    yes_re = re.compile('yes="(.*?)"')
    quoted_re = re.compile('"(.*?)"')
    instances = text_re.findall(f_content)

    dict_instances = []
    for instance in instances:
        id_line, f_line, m_line, c_line, content_line, _ = instance.split("\n")
        line = dict(fine=f_line, medium=m_line, coarse=c_line)[granularity]

        id_no = id_re.search(id_line).group(1)
        interval_str = yes_re.search(line).group(1)
        interval = _interval_from_str(interval_str)

        possible_intervals = map(_interval_from_str, quoted_re.findall(line))
        dict_instances.append(dict(id=id_no, true_interval=interval,
                                   possible_intervals=possible_intervals,
                                   text=content_line))
    return dict_instances


if __name__ == '__main__':
    import sys
    import json
    import io

    try:
        infile = sys.argv[1]
    except IndexError:
        print("Usage: python {} path/to/training08Tx.txt > "
              "path/to/output.json".format(sys.argv[0]))
        sys.exit(1)

    with io.open(infile) as f:
        instances = read_semeval_data(f.read())
    print(json.dumps(instances))
