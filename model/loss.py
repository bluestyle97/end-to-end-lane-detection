import torch
import torch.nn as nn
import torch.nn.functional as F


class LanenetLoss(object):
    def __init__(self):
        self.device = torch.device('cpu')

    @staticmethod
    def unique_with_counts(input):
        '''
        :param input: 1-D tensor.
        :return: unique elements, unique id, counts
        '''
        unique_elements, unique_id = input.unique(sorted=True, return_inverse=True)
        unique_elements_num = unique_elements.numel()
        origin_elements_num = input.numel()
        inp_repeat = input.view(1, origin_elements_num).repeat(unique_elements_num, 1)
        ele_repeat = unique_elements.view(unique_elements_num, 1).repeat(1, origin_elements_num)
        counts = torch.sum(inp_repeat==ele_repeat, dim=1).view(unique_elements_num).float()
        return unique_elements, unique_id, counts

    @staticmethod
    def unsorted_segment_sum(data, segment_ids, num_segments):
        assert data.size()[0] == segment_ids.size()[0]

        segment_sum = torch.zeros(num_segments, data.size()[1], dtype=data.dtype)
        for i in range(data.size()[0]):
            segment_sum[segment_ids[i], :] += data[i, :]
        return segment_sum

    def cross_entropy_loss_single(self, pred, label):
        '''
        :param pred: 2xHxW tensor.
        :param label: 1xHxW tensor.
        '''
        unique_labels, unique_id, counts = self.unique_with_counts(label.view(label.numel()))
        inverse_weights = torch.div(
            torch.tensor(1.0, dtype=torch.float32),
            torch.log(torch.add(torch.div(torch.tensor(1.0, dtype=torch.float32), counts), 1.02)))

        loss_fn = nn.CrossEntropyLoss(weight=inverse_weights)

        height, width = pred.size()[-2], pred.size()[-1]
        pred = pred.view(1, 2, height, width)
        label = label.view(1, height, width).to(torch.long)

        loss = loss_fn(pred, label)
        return loss

    def cross_entropy_loss(self, pred, label):
        '''
        :param pred: Bx2xHxW tensor.
        :param label: Bx1xHxW tensor.
        '''
        batch_size = pred.size()[0]
        loss_acc = torch.zeros(1, dtype=torch.float32)

        for i in range(batch_size):
            loss = self.cross_entropy_loss_single(pred[i], label[i])
            loss_acc += loss

        loss = torch.div(loss_acc, batch_size).squeeze()
        return loss

    def discriminative_loss_single(self,
            prediction,
            label,
            feature_dim,
            delta_v,
            delta_d,
            param_var,
            param_dist,
            param_reg):
        label = label.view(label.numel())
        pred = prediction.transpose(0, 2).contiguous().view(-1, feature_dim)

        unique_labels, unique_id, counts = self.unique_with_counts(label)
        num_instances = torch.numel(unique_labels)

        # compute mean vector of pixel embedding
        segmented_sum = self.unsorted_segment_sum(pred, unique_id, num_instances)
        mu = torch.div(segmented_sum, counts.view(-1, 1))
        idx = unique_id.view(-1, 1).repeat(1, mu.size()[1])
        mu_expand = torch.gather(mu, dim=0, index=idx)

        # compute loss(var)
        distance = torch.norm(mu_expand - pred, p=2, dim=1, keepdim=True)
        distance = F.relu(distance - delta_v)
        distance = torch.pow(distance, 2)

        l_var = self.unsorted_segment_sum(distance, unique_id, num_instances)
        l_var = torch.div(l_var, counts.view(-1, 1))
        l_var = torch.sum(l_var)
        l_var = torch.div(l_var, float(num_instances))

        # compute loss(dist)
        mu_interleaved_rep = mu.repeat(num_instances, 1)
        mu_band_rep = mu.repeat(1, num_instances).view(num_instances * num_instances, feature_dim)
        mu_diff = mu_band_rep - mu_interleaved_rep

        intermediate_tensor = torch.sum(torch.abs(mu_diff), dim=1, keepdim=True)
        bool_mask = torch.eq(intermediate_tensor, torch.zeros(1, dtype=torch.float32)).view(
            intermediate_tensor.size()[0])
        mu_diff_bool = mu_diff[bool_mask == 0, :]

        mu_norm = torch.norm(mu_diff_bool, p=2, dim=1, keepdim=True)
        mu_norm = 2.0 * delta_d - mu_norm
        mu_norm = F.relu(mu_norm)
        mu_norm = torch.pow(mu_norm, 2)

        l_dist = torch.mean(mu_norm, dim=0).squeeze()

        # compute regularization loss proposed in the original discriminative loss paper
        l_reg = torch.mean(torch.norm(mu, p=2, dim=1))

        # merge all losses
        param_scale = 1.
        l_var = param_var * l_var
        l_dist = param_dist * l_dist
        l_reg = param_reg * l_reg

        loss = param_scale * (l_var + l_dist + l_reg)

        return loss, l_var, l_dist, l_reg

    def discriminative_loss(self,
            pred,
            label,
            feature_dim,
            delta_v,
            delta_d,
            param_var,
            param_dist,
            param_reg):
        batch_size = pred.size()[0]

        loss_acc = torch.zeros(batch_size, dtype=torch.float32)
        l_var_acc = torch.zeros(batch_size, dtype=torch.float32)
        l_dist_acc = torch.zeros(batch_size, dtype=torch.float32)
        l_reg_acc = torch.zeros(batch_size, dtype=torch.float32)

        for i in range(batch_size):
            loss_acc[i], l_var_acc[i], l_dist_acc[i], l_reg_acc[i] = self.discriminative_loss_single(pred[i], label[i], feature_dim,
                                                                             delta_v, delta_d, param_var,
                                                                             param_dist, param_reg)

        loss = torch.mean(loss_acc)
        l_var = torch.mean(l_var_acc)
        l_dist = torch.mean(l_dist_acc)
        l_reg = torch.mean(l_reg_acc)

        return loss, l_var, l_dist, l_reg


class HNetLoss(object):
    @staticmethod
    def hnet_loss_single(pts_gt, trans_coef):
        pts_gt = pts_gt.view(-1, 3)
        trans_coef = trans_coef.view(6)
        trans_coef = torch.cat([trans_coef, torch.tensor([1.0])])
        H_indices = torch.tensor([0, 1, 2, 4, 5, 7, 8], dtype=torch.long)
        H = torch.zeros(9, dtype=torch.float32).scatter_(0, H_indices, trans_coef).view(3, 3)

        pts_gt = pts_gt.view(-1, 3).to(torch.float32).t()  # (3 * n)
        pts_projected = torch.mm(H, pts_gt)

        # least squares closed-form solution
        X = pts_projected[0, :].view(-1, 1)  # (n * 1)
        Y = pts_projected[1, :].view(-1, 1)  # (n * 1)
        Y_mat = torch.cat([torch.pow(Y, 3), torch.pow(Y, 2), Y, torch.ones_like(Y, dtype=torch.float32)],
                          dim=1)  # (n * 4)
        w = Y_mat.t().mm(Y_mat).inverse().mm(Y_mat.t()).mm(X)  # (4 * 1)

        # re-projection and compute loss

        x_pred = torch.mm(Y_mat, w)  # (n * 1)
        pts_pred = torch.cat([x_pred, Y, torch.ones_like(Y, dtype=torch.float32)], dim=1).t()  # (3 * n)
        pts_back = torch.mm(H.inverse(), pts_pred)

        loss = torch.mean(torch.pow(pts_back[0, :] - pts_gt[0, :], 2))

        return loss

    def hnet_loss(self, pts_batch, coef_batch):
        batch_size = coef_batch.size()[0]

        loss_acc = torch.zeros(batch_size, dtype=torch.float32)

        for i in range(batch_size):
            loss_acc[i] = self.hnet_loss_single(pts_batch[i], coef_batch[i])

        loss = torch.mean(loss_acc)
        return loss


if __name__ == '__main__':
    lanenet_loss = LanenetLoss()
    pred = torch.rand(4, 2, 4, 4, dtype=torch.float32)
    label = torch.randint(3, (4, 1, 4, 4), dtype=torch.long)
    feature_dim = 2
    delta_v = 0.5
    delta_d = 3.0
    param_val = 1.0
    param_dist = 1.0
    param_reg = 0.001
    loss, l_val, l_dist, l_reg = lanenet_loss.discriminative_loss(pred, label, feature_dim, delta_v, delta_d, param_val, param_dist, param_reg)
    print(loss, l_val, l_dist, l_reg)

    hnet_loss = HNetLoss()
    pts = torch.rand(4, 10, 3, dtype=torch.float32)
    coef = torch.rand(4, 6, dtype=torch.float32)
    loss = hnet_loss.hnet_loss(pts, coef)
    print(loss)