import torch
import torch.nn as nn

def passing_normalize_argument(args) :
    global argument
    argument = args
class NormalActivator(nn.Module):

    def __init__(self, loss_focal, loss_l2, use_focal_loss):
        super(NormalActivator, self).__init__()

        self.do_normalized_score = argument.do_normalized_score

        # [1]
        self.anomal_feat_list = []
        self.normal_feat_list = []

        # [2]
        self.attention_loss = {}
        self.attention_loss['normal_cls_loss'] = []
        self.attention_loss['anormal_cls_loss'] = []
        self.attention_loss['normal_trigger_loss'] = []
        self.attention_loss['anormal_trigger_loss'] = []
        self.trigger_score = []
        self.cls_score = []

        # [3]
        self.loss_focal = loss_focal
        self.loss_l2 = loss_l2
        self.anomal_map_loss = []
        self.use_focal_loss = use_focal_loss

        # [4]
        self.normal_matching_query_loss = []
        self.resized_queries = []
        self.keys = []

    def collect_queries(self, origin_query, anomal_position_vector, do_collect_normal = True):

        pix_num = origin_query.shape[0]
        for pix_idx in range(pix_num):
            feat = origin_query[pix_idx].squeeze(0)
            anomal_flag = anomal_position_vector[pix_idx]
            if anomal_flag == 1:
                self.anomal_feat_list.append(feat.unsqueeze(0))
            else:
                if do_collect_normal:
                    self.normal_feat_list.append(feat.unsqueeze(0))

    def collect_attention_scores(self, attn_score, anomal_position_vector,
                                 do_normal_activating = True):

        def normalize_score(score):
            score = torch.softmax(score, dim=-1)
            max_value = (torch.max(score, dim=-1)[0]).unsqueeze(-1)
            normalized_trigger_map = score / max_value
            return score

        # [1] preprocessing
        cls_score, trigger_score = attn_score.chunk(2, dim=-1)
        cls_score, trigger_score = cls_score.squeeze(), trigger_score.squeeze()      # head, pix_num

        if self.do_normalized_score :
            cls_score, trigger_score = normalize_score(cls_score), normalize_score(trigger_score)

        cls_score, trigger_score = cls_score.mean(dim=0), trigger_score.mean(dim=0)  # pix_num
        total_score = torch.ones_like(cls_score)

        # [2]
        normal_cls_score = cls_score * (1 - anomal_position_vector)
        normal_trigger_score = trigger_score * (1 - anomal_position_vector)
        anomal_cls_score = cls_score * anomal_position_vector
        anomal_trigger_score = trigger_score * anomal_position_vector


        # [3]
        normal_cls_score = normal_cls_score / total_score
        normal_trigger_score = normal_trigger_score / total_score
        anomal_cls_score = anomal_cls_score / total_score
        anomal_trigger_score = anomal_trigger_score / total_score

        # [4]
        normal_cls_loss = normal_cls_score ** 2
        normal_trigger_loss = (1 - normal_trigger_score ** 2 )  #normal cls score 이랑 같은 상황
        anomal_cls_loss = (1 - anomal_cls_score ** 2)
        anomal_trigger_loss = anomal_trigger_score ** 2

        # [5]
        if do_normal_activating:
            self.attention_loss['normal_cls_loss'].append(normal_cls_loss.mean())
            self.attention_loss['normal_trigger_loss'].append(normal_trigger_loss.mean())

        anomal_pixel_num = anomal_position_vector.sum()
        if anomal_pixel_num > 0:
            self.attention_loss['anormal_cls_loss'].append(anomal_cls_loss.mean())
            self.attention_loss['anormal_trigger_loss'].append(anomal_trigger_loss.mean())

    def collect_anomal_map_loss(self, attn_score, anomal_position_vector):

        if self.use_focal_loss:

            cls_score, trigger_score = attn_score.chunk(2, dim=-1)
            res = int(cls_score.shape[1] ** 0.5)
            head_num = cls_score.shape[0]
            cls_score = cls_score.view(-1, res, res).unsqueeze(1) # [8,1,64,64]
            trigger_score = trigger_score.view(-1, res, res).unsqueeze(1) # [8,1,64,64]
            focal_loss_in = torch.cat([cls_score,trigger_score], 1) # [8,2,64,64]

            # [2] target
            focal_loss_trg = anomal_position_vector.view(res, res).unsqueeze(0).unsqueeze(0).repeat(head_num,1,1,1)
            map_loss = self.loss_focal(focal_loss_in,
                                       focal_loss_trg.to(dtype=trigger_score.dtype))

        else:
            cls_score, trigger_score = attn_score.chunk(2, dim=-1)
            cls_score, trigger_score = cls_score.squeeze(), trigger_score.squeeze()      # head, pix_num
            cls_score, trigger_score = cls_score.mean(dim=0), trigger_score.mean(dim=0)  # pix_num
            trg_trigger_score = 1 - anomal_position_vector
            map_loss = self.loss_l2(trigger_score.float(), trg_trigger_score.float())


        self.anomal_map_loss.append(map_loss)

    def generate_mahalanobis_distance_loss(self):

        def mahal(u, v, cov):
            delta = u - v
            m = torch.dot(delta, torch.matmul(cov, delta))
            return torch.sqrt(m)

        normal_feats = torch.cat(self.normal_feat_list, dim=0)
        mu = torch.mean(normal_feats, dim=0)
        cov = torch.cov(normal_feats.transpose(0, 1))
        normal_mahalanobis_dists = [mahal(feat, mu, cov) for feat in normal_feats]
        normal_dist_max = torch.tensor(normal_mahalanobis_dists).max()
        normal_dist_mean = torch.tensor(normal_mahalanobis_dists).mean()

        if len(self.anomal_feat_list) > 0 :
            anormal_feats = torch.cat(self.anomal_feat_list, dim=0)
            anormal_mahalanobis_dists = [mahal(feat, mu, cov) for feat in anormal_feats]
            anormal_dist_mean = torch.tensor(anormal_mahalanobis_dists).mean()
            total_dist = normal_dist_mean + anormal_dist_mean
            normal_dist_loss = (normal_dist_mean / total_dist).requires_grad_()
        else :
            normal_dist_loss = normal_dist_max.requires_grad_()
        self.normal_feat_list = []
        self.anomal_feat_list = []
        return normal_dist_loss, normal_dist_mean, normal_dist_max

    def generate_attention_loss(self):

        normal_cls_loss = 0.0
        normal_trigger_loss = 0.0
        if len(self.attention_loss['normal_cls_loss']) != 0:
            normal_cls_loss = torch.stack(self.attention_loss['normal_cls_loss'], dim=0).mean(dim=0)
            normal_trigger_loss = torch.stack(self.attention_loss['normal_trigger_loss'], dim=0).mean(dim=0)

        anormal_cls_loss = 0.0
        anormal_trigger_loss = 0.0
        if len(self.attention_loss['anormal_cls_loss']) != 0:
            anormal_cls_loss = torch.stack(self.attention_loss['anormal_cls_loss'], dim=0).mean(dim=0)
            anormal_trigger_loss = torch.stack(self.attention_loss['anormal_trigger_loss'], dim=0).mean(dim=0)


        self.attention_loss = {'normal_cls_loss': [], 'normal_trigger_loss': [],
                               'anormal_cls_loss': [], 'anormal_trigger_loss': []}
        return normal_cls_loss, normal_trigger_loss, anormal_cls_loss, anormal_trigger_loss
    def generate_anomal_map_loss(self):
        map_loss = torch.stack(self.anomal_map_loss, dim=0)
        map_loss = map_loss.mean()
        self.anomal_map_loss = []
        return map_loss

    def matching_TS_query(self, T_query, S_query, anomal_position_vector):
        pix_num, dim = T_query.shape
        anomal_position_matrix = anomal_position_vector.unsqueeze(-1).repeat(1, dim)
        T_query = T_query * (1 - anomal_position_matrix)
        S_query = S_query * (1 - anomal_position_matrix)
        matching_query_loss = self.loss_l2(T_query.float(), S_query.float())
        self.normal_matching_query_loss.append(matching_query_loss)

    def collect_qk_features(self, query, key, anomal_position_vector, do_collect_normal = True) :

        head_num, pix_num, dim = query.shape
        res = int(pix_num ** 0.5)
        query_map = query.view(head_num, res, res, dim).permute(0, 3, 1, 2).contiguous()
        resized_query_map = nn.functional.interpolate(query_map, size=(64, 64), mode='bilinear')
        resized_query = resized_query_map.permute(0, 2, 3, 1).contiguous().view(head_num, -1, dim)  # head, 64*64, dim

        self.collect_queries(resized_query.mean(dim=0),
                             anomal_position_vector,
                             do_collect_normal = do_collect_normal)
        self.resized_queries.append(resized_query)
        self.keys.append(key)

    def generate_conjugated_attention(self,
                                      anomal_position_vector, do_normal_activating):

        concat_query = torch.cat(self.resized_queries, dim=2)  # 8, 4096, 960
        concat_key = torch.cat(self.keys, dim=2)  # 8, 77, 960
        attn_score = torch.bmm(concat_query, concat_key.permute(0, 2, 1))  # 8, 4096, 77
        self.resized_queries = []
        self.keys = []

        self.collect_attention_scores(attn_score,
                                      anomal_position_vector,
                                      do_normal_activating = do_normal_activating)
        return attn_score


    def reset(self) -> None:

        # [1]
        self.anomal_feat_list = []
        self.normal_feat_list = []

        # [2]
        self.attention_loss = {'normal_cls_loss': [], 'normal_trigger_loss': [],
                               'anormal_cls_loss': [], 'anormal_trigger_loss': []}
        self.trigger_score = []
        self.cls_score = []

        # [3]
        self.anomal_map_loss = []

        # [4]
        self.normal_matching_query_loss = []
        self.resized_queries = []
        self.keys = []