import torch
import dataset
import stats
import os

# root = "/private/home/yuandong/forked/luckmatters/catalyst/outputs/2020-02-25/16-00-39"

models = dict(
    no_adv = "/private/home/yuandong/forked/luckmatters/catalyst/outputs/2020-02-27/21-41-17",
    adv = "/private/home/yuandong/forked/luckmatters/catalyst/outputs/2020-02-28/10-16-54"
)

for key, root in models.items():
    print("====== " + key + f": {root}")
    teacher = torch.load(os.path.join(root, "teacher-0.pt"))
    student = torch.load(os.path.join(root, "student-90.pt"))

    teacher.cuda()
    student.cuda()

    teacher.eval()
    student.eval()

    stats_weight_corr = stats.WeightCorr(teacher, student)

    print("Weight corr:")
    stats_weight_corr.export()
    print(stats_weight_corr.prompt()[0])

    data = dataset.RandomDataset(10240, (500,), 10)
    dataset_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True, num_workers=4)

    stats_corr = stats.StatsCorr(teacher, student)

    for x, label in dataset_loader:
        x, label = x.cuda(), label.cuda()

        output_t = teacher(x)
        output_s = student(x)

        stats_corr.add(output_t, output_s, label)

    print("State corr:")
    stats_corr.export()

    summary, data = stats_corr.prompt(verbose=True)
    print(summary)

    print("Data")
    print(data)
