import torch
import dataset
import stats
import os

import utils_ as utils

# root = "/private/home/yuandong/forked/luckmatters/catalyst/outputs/2020-02-25/16-00-39"

models = dict(
    # no_adv = "/private/home/yuandong/forked/luckmatters/catalyst/outputs/2020-02-27/21-41-17",
    # adv = "/private/home/yuandong/forked/luckmatters/catalyst/outputs/2020-02-28/10-16-54",
    # three_layer = "/private/home/yuandong/forked/luckmatters/catalyst/outputs/2020-02-29/08-14-41",
    no_adv2 = "/private/home/yuandong/forked/luckmatters/catalyst/outputs/2020-02-29/10-00-34",
)

for key, root in models.items():
    print("====== " + key + f": {root}")
    teacher = torch.load(os.path.join(root, "teacher-0.pt"))
    student = torch.load(os.path.join(root, "student-49.pt"))

    train_dataset = os.path.join(root, "train_dataset.pth")
    eval_dataset = os.path.join(root, "eval_dataset.pth")

    if os.path.exists(train_dataset):
        print("Loading dataset.")
        data = torch.load(train_dataset)
    else:
        data = dataset.RandomDataset(10240, (500,), 10)

    teacher.cuda()
    student.cuda()

    teacher.eval()
    student.eval()

    stats_weight_corr = stats.WeightCorr(teacher, student)

    print("Weight corr:")
    stats_weight_corr.export()
    summary = stats_weight_corr.prompt()
    print(summary["summary"])

    dataset_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True, num_workers=4)

    stats_corr = stats.StatsCorr(teacher, student)

    for x, label in dataset_loader:
        x, label = x.cuda(), label.cuda()

        output_t = teacher(x)
        output_s = student(x)

        stats_corr.add(output_t, output_s, label)

    print("State corr:")
    stats_corr.export()

    summary = stats_corr.prompt(verbose=True)
    print(summary["summary"])

    print("Data")
    print(summary["data"])

    # Compute SVD for the first output. 
    outputs = utils.concatOutput(dataset_loader, [teacher, student])

    layer_idx = 0

    h_t = outputs[0]["hs"][layer_idx]
    h_s = outputs[1]["hs"][layer_idx]

    student_sel = summary["best_students"][layer_idx]
    h_s = h_s[:, student_sel]

    print("Size of h_t: ", h_t.size())
    print("Size of h_s: ", h_s.size())

    _, s_t, _ = torch.svd(h_t)
    _, s_s, _ = torch.svd(h_s)
    
    print(f"Normalized singular value for space of teacher at layer {layer_idx}")
    print(s_t / s_t[0])
    print(f"Normalized singular value for space of student at layer {layer_idx}")
    print(s_s / s_s[0])

    
