import torch
import basic_tools 

def tune_teacher(data_loader, teacher):
    # Tune the bias of the teacher so that their activation/inactivation is approximated 0.5/0.5
    num_hidden = teacher.num_hidden_layers()
    for t in range(num_hidden):
        output = basic_tools.utils.concatOutput(data_loader, [teacher])

        act = output[0]["post_lins"][t]

        if act.dim() == 4:
            act = act.permute(0, 2, 3, 1).reshape(-1, act.size(1))

        estimated_bias = act.median(dim=0)[0]
        teacher.ws_linear[t].bias.data[:] -= estimated_bias.cuda() 


def tune_teacher_last_layer(data_loader, teacher):
    output = basic_tools.utils.concatOutput(data_loader, [teacher])

    # Tune the final linear layer to make output balanced as well. 
    y = output[0]["y"]
    y_mean = y.mean(dim=0).cuda()
    y_std = y.std(dim=0).cuda()
    
    teacher.final_w.weight.data /= y_std[:, None]
    teacher.final_w.bias.data -= y_mean
    teacher.final_w.bias.data /= y_std


def check(data_loader, teacher, output_func=print):
    # double check
    output = basic_tools.utils.concatOutput(data_loader, [teacher])[0]
    num_hidden = teacher.num_hidden_layers()
    for t in range(num_hidden):
        margin = output["post_lins"][t]
        activate_ratio = (margin > 0).float().mean(dim=0)
        margin_std = margin.std(dim=0)
        output_func(f"{t}: activate_ratio: {activate_ratio}")
        output_func(f"{t}: margin_std: {margin_std}")

    y = output["y"]
    y_mean = y.mean(dim=0)
    y_std = y.std(dim=0)
    output_func(f"Final layer: y_mean: {y_mean}, y_std: {y_std}")

