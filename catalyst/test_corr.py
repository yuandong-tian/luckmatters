import stats_operator
import torch
import utils
import utils_corrs
import vis_corrs 
from argparse import Namespace

def genData(batchsize, dim):
    hs = []
    for i in range(4):
        hs.append(torch.FloatTensor(batchsize, dim).uniform_())

    return hs

def getSubset(hs, idx, idx_end):
    return [ h[idx:idx_end, :] for h in hs ]


def testCorr():
    ''' Test old and new correlation '''
    corr_collector = stats_operator.StatsCorr(None, None)
    corr_collector.reset()

    bs = 64
    num_batch = 100
    N = bs * num_batch

    teacher_d = 20
    student_d = 40

    teacher_hs = genData(N, teacher_d)
    student_hs = genData(N, student_d)

    for i in range(num_batch):
        o_t = dict(hs=getSubset(teacher_hs, i * bs, (i + 1) * bs))
        o_s = dict(hs=getSubset(student_hs, i * bs, (i + 1) * bs))

        corr_collector.add(o_t, o_s, None)

    corrs = utils_corrs.acts2corrIndices(teacher_hs, student_hs)
    print("Summary from old version of corr: ")
    print(vis_corrs.get_corrs(corrs))

    print("Summary from new version of corr: ")
    corr_collector.export()
    print(corr_collector.prompt())


if __name__ == "__main__":
    testCorr()
