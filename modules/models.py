import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class Classifier(nn.Module):
    def __init__(self, num_keywords, num_category, reduced_feature):
        super().__init__()
        self.num_category = num_category
        self.num_keywords = num_keywords
        self.reduced_feature = reduced_feature

        layers = [(num_keywords*reduced_feature, 1024), (1024, 512), (512, 256)]
        self.linear = nn.ModuleList([nn.Linear(i, o) for i, o in layers])
        self.bn = nn.ModuleList([nn.BatchNorm1d(o) for i, o in layers])

        self.act = nn.GELU()
        self.last = nn.Linear(256, num_category)
        
    def forward(self, x):
        x = x.view(-1, self.num_keywords*self.reduced_feature)
        batch_size = x.shape[0]
        for lin, bn in zip(self.linear, self.bn):
            if batch_size > 1: bn.train()
            else: bn.eval()
            x = bn(self.act(lin(x)))
        x = self.last(x)
        return x


def criterion(preds, labels, is_augmented_list, args):
    original_loss = F.cross_entropy(preds, labels, reduction='none', label_smoothing=0)
    label_smoothed_loss = F.cross_entropy(preds, labels, reduction='none', label_smoothing=args['soft_label'])
    return torch.mean(torch.where(is_augmented_list, label_smoothed_loss, original_loss))



class WordCritic(nn.Module):
    def __init__(self, num_keywords, num_category, feature_size, latent_size):
        super().__init__()
        self.num_category = num_category
        self.num_keywords = num_keywords
        self.feature_size = feature_size
        self.latent_size = latent_size


        linear = [(self.latent_size, 512), (512, 256), (256, 128), (128, 64)]
        self.linear = nn.ModuleList([nn.Linear(i, o) for i, o in linear])
        self.linear_bn = nn.ModuleList([nn.BatchNorm1d(o) for i, o in linear])

        self.last= nn.Linear(64, 1)
        self.drop = nn.Dropout(0.2)
        self.act = nn.GELU()


    def forward(self, x):
        batch_size = x.shape[0]

        x = x.view(-1, self.latent_size)
        for lin, bn in zip(self.linear, self.linear_bn):
            if batch_size > 1: bn.train()
            else: bn.eval()
            x = self.drop(self.act(bn(lin(x))))

        x = self.last(x)
        return x

class WordEncoder(nn.Module):
    def __init__(self, num_keywords, num_category, feature_size, order_encoding, base_cuboid, latent_size):
        super().__init__()
        self.num_category = num_category
        self.num_keywords = num_keywords
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.order = order_encoding.cuda()
        order_size = 128
        
        style = 32

        down_linear = [(self.feature_size+style, 1024), (1024+style, 1024), (1024+style, 512), (512+style, 256), (256+style, 256)]
        self.down_linear = nn.ModuleList([nn.Linear(i, o) for i, o in down_linear])
        self.down_linear_bn = nn.ModuleList([nn.BatchNorm1d(o) for i, o in down_linear])

        self.last_embedding = nn.Linear(order_size+num_category, style)
        self.embedding = nn.ModuleList([nn.Linear(order_size+num_category, style) for _, _ in down_linear])
        self.latent_last= nn.Linear(256+style, self.latent_size)
        self.act = nn.GELU()

        self.base_cuboid = base_cuboid.cuda()

    def forward(self, x, label, true_samples=True):
        batch_size = x.shape[0]
        x = x - self.base_cuboid
        order = torch.repeat_interleave(self.order, batch_size, 0)
        x = x.view(batch_size, self.num_keywords, self.feature_size)
        x = x.view(-1, self.latent_size)
        one_hot = F.one_hot(label, self.num_category).cuda()
        one_hot = torch.repeat_interleave(one_hot.unsqueeze(1), self.num_keywords).view(-1, self.num_category)
        style = torch.cat((order, one_hot), 1)

        for lin, bn, emb in zip(self.down_linear, self.down_linear_bn, self.embedding):
            if batch_size > 1 and true_samples: bn.train()
            else: bn.eval()
            embedding = emb(style)
            x = torch.cat((x, embedding), 1)
            x = self.act(bn(lin(x)))

        embedding = self.last_embedding(style)
        latent = self.latent_last(torch.cat((x, embedding), 1))
        latent = latent.view(batch_size, self.num_keywords, self.latent_size)
        return latent

class WordDecoder(nn.Module):
    def __init__(self, num_keywords, num_category, feature_size, order_encoding, base_cuboid, latent_size):
        super().__init__()
        self.num_category = num_category
        self.num_keywords = num_keywords
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.order = order_encoding.cuda()
        order_size = 128

        style = 32

        up_linear = [(latent_size+style, 256), (256+style, 512), (512+style, 512), (512+style, 1024), (1024+style, 1024)]
        self.up_linear = nn.ModuleList([nn.Linear(i, o) for i, o in up_linear])
        self.up_linear_bn = nn.ModuleList([nn.BatchNorm1d(o) for i, o in up_linear])

        self.last= nn.Linear(1024+style, self.feature_size)
        self.last_embedding = nn.Linear(order_size+num_category, style)
        self.act = nn.GELU()

        self.embedding = nn.ModuleList([nn.Linear(order_size+num_category, style) for _, _ in up_linear])

        self.base_cuboid = base_cuboid.cuda()

    def forward(self, x, label):
        batch_size = x.shape[0]
        order = torch.repeat_interleave(self.order, batch_size, 0)
        x = x.view(-1, self.latent_size)
        one_hot = F.one_hot(label, self.num_category).cuda()
        one_hot = torch.repeat_interleave(one_hot.unsqueeze(1), self.num_keywords).view(-1, self.num_category)
        style = torch.cat((order, one_hot), 1)
        
        for lin, bn, emb in zip(self.up_linear, self.up_linear_bn, self.embedding):
            if batch_size > 1: bn.train()
            else: bn.eval()
            embedding = emb(style)
            x = torch.cat((x, embedding), 1)
            x = self.act(bn(lin(x)))

        embedding = self.last_embedding(style)
        x = self.last(torch.cat((x, embedding), 1))
        x = x.view(batch_size, self.num_keywords, self.feature_size)
        x = x + self.base_cuboid
        return x



class CriticTail(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_size = 128

        self.out = nn.Linear(self.latent_size, 1)
        
    def forward(self, x):
        score = self.out(x)
        
        return score

class Generator(nn.Module):
    def __init__(self, num_keywords, num_category, feature_size):
        super().__init__()
        self.num_category = num_category
        self.num_keywords = num_keywords
        self.feature_size = feature_size
        self.latent_size = 512
        self.latent_word = 2

        up_linear = [(self.latent_size, 512), (512, 2048), (2048, self.latent_word*num_keywords)]
        self.up_linear = nn.ModuleList([nn.Linear(i, o) for i, o in up_linear])
        self.up_linear_bn = nn.ModuleList([nn.BatchNorm1d(o) for i, o in up_linear])

        self.word_up_linear = nn.ModuleList([nn.Linear(self.latent_word, 512), nn.Linear(512, feature_size)])
        self.word_up_bn = nn.ModuleList([nn.BatchNorm1d(512), nn.BatchNorm1d(feature_size)])

        self.last= nn.Linear(feature_size, feature_size)
        self.drop = nn.Dropout(0.5)
        self.act = nn.GELU()

    def forward(self, label):
        batch_size = label.shape[0]
        x = torch.normal(0, 1, size=(batch_size, self.latent_size), device='cuda')
        for lin, bn in zip(self.up_linear, self.up_linear_bn):
            if batch_size > 1: bn.train()
            else: bn.eval()
            x = self.act(bn(lin(x)))

        x = x.view(batch_size*self.num_keywords, -1)
        for lin, bn in zip(self.word_up_linear, self.word_up_bn):
            if batch_size > 1: bn.train()
            else: bn.eval()
            x = self.act(bn(lin(x)))

        x = self.drop(x)
        x = self.last(x)

        x = x.view(batch_size, self.num_keywords, self.feature_size)
        norm = torch.linalg.norm(x, ord=2, dim=2, keepdim=True)
        x = (torch.where(torch.repeat_interleave(norm>10, self.feature_size, dim=2), x, .0) - x).detach() + x
        return x





# class _Generator(nn.Module):
#     def __init__(self, num_keywords, num_category, feature_size):
#         super().__init__()
#         self.num_category = num_category
#         self.num_keywords = num_keywords
#         self.feature_size = feature_size
#         self.random_start = 128
#         self.latent = 2
#         self.random_word = 64

#         self.inital_latent = nn.Linear(self.random_start+self.num_category, 128*4)

#         latent_conv = [
#             (128, 64, 4, 4),
#             (64, 32, 5, 5),
#             (32, 8, 5, 5),
#             (8, 4, 5, 5),
#             (4, 1, 5, 5)
#         ]
#         self.latent_conv = nn.ModuleList([nn.ConvTranspose1d(i, o, k, s) for i, o, k, s in latent_conv])
#         self.latent_bn = nn.ModuleList([nn.BatchNorm1d(o) for i, o, k, s in latent_conv])

#         self.latent_out = 4
#         for i, o, k, s in latent_conv:
#             self.latent_out = (self.latent_out-1)*s + (k-1) + 1
#         self.latent_out = self.latent_out * latent_conv[-1][1]


#         word_conv = [
#             (self.latent+self.random_word, 32, 6, 6),
#             (32, 16, 5, 5),
#             (16, 4, 4, 4),
#             (4, 1, 3, 3)
#         ]
#         self.word_conv = nn.ModuleList([nn.ConvTranspose1d(i, o, k, s) for i, o, k, s in word_conv])
#         self.word_bn = nn.ModuleList([nn.BatchNorm1d(o) for i, o, k, s in word_conv])

#         self.word_out = 1
#         for i, o, k, s in word_conv:
#             self.word_out = (self.word_out-1)*s + (k-1) + 1
#         self.word_out = self.word_out * word_conv[-1][1]
        
#         # latent_layers = [((self.random_start+self.num_category), 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, num_keywords)]
#         latent_last_layer = (self.latent_out, self.latent*num_keywords)
#         selection_layer = (self.latent_out, num_keywords)

#         # word_generator_layers = [(self.latent+self.random_word, 64), (64, 128), (128, 256), (256, 512)]
        
#         word_last_layer = (self.word_out, feature_size)

#         # self.latent_linear = nn.ModuleList([nn.Linear(i, o) for i, o in latent_layers])
#         # self.latent_bn = nn.ModuleList([nn.BatchNorm1d(o) for i, o in latent_layers])
#         self.dropout = nn.Dropout1d(0.25)

#         self.latent_last = nn.Linear(*latent_last_layer)
#         self.selection = nn.Linear(*selection_layer)

#         # word_linear = []
#         # for i, o in word_generator_layers:
#         #     word_linear.append(nn.Linear(i, o))
#         # self.word_linear = nn.ModuleList(word_linear)

#         self.word_last = nn.Linear(*word_last_layer)
#         self.act = nn.GELU()
#         self.pe = PositionalEncoding(self.latent+self.random_word, self.num_keywords, 'cuda')

#     def forward(self, label):
#         batch_size = label.shape[0]

#         one_hot = F.one_hot(label, num_classes=self.num_category)
#         start_random = torch.normal(0, 1, size=(batch_size, self.random_start), device='cuda')
#         x = torch.cat((start_random, one_hot), dim=1)
#         x = self.inital_latent(x)

#         x = x.view(batch_size, 128, 4) # for conv
#         for conv, bn in zip(self.latent_conv, self.latent_bn):
#             if batch_size > 1: bn.train()
#             else: bn.eval()
#             x = self.dropout(self.act(bn(conv(x))))

#         selection = F.leaky_relu(self.selection(x).view(batch_size*self.num_keywords, 1)) # -alpha ~ 
#         # selection = (torch.round(selection) - selection).detach() + selection # rounding with STE

#         x = self.latent_last(x) # keywords * latent

#         x = x.view(batch_size*self.num_keywords, self.latent)
#         rand_latent = torch.normal(0, 1, size=(batch_size*self.num_keywords, self.random_word), device='cuda')
#         x = torch.cat((x, rand_latent), dim=1)

#         x = x.view(batch_size, self.num_keywords, self.latent+self.random_word)
#         x += self.pe(x)

#         x = x.view(batch_size*self.num_keywords, self.latent+self.random_word, 1)
#         for conv, bn in zip(self.word_conv, self.word_bn):
#             if batch_size > 1: bn.train()
#             else: bn.eval()
#             x = self.dropout(self.act(bn(conv(x))))

#         x = x.view(batch_size*self.num_keywords, -1)
#         x = F.tanh(self.word_last(x)) * selection

#         x = x.view(batch_size, self.num_keywords, self.feature_size)
#         return x, selection

# class _Critic(nn.Module):
#     def __init__(self, num_keywords, num_category, feature_size):
#         super().__init__()
#         self.num_category = num_category
#         self.num_keywords = num_keywords
#         self.feature_size = feature_size
#         col_conv = [
#             (1, 4, (1, 8), (1, 8)), 
#             (4, 8, (1, 8), (1, 8)), 
#             (8, 16, (1, 2), (1, 2)), 
#         ]
#         row_conv = [
#             (16, 32, (2, 1), (2, 1)),
#             (32, 32, (5, 1), (3, 1)),
#             (32, 64, (5, 1), (3, 1)),
#             (64, 64, (5, 1), (3, 1)),
#             (64, 128, (5, 1), (3, 1)),
#             (128, 128, (5, 1), (3, 1))
#         ]

#         output_size = [num_keywords, feature_size]
#         for _, _, k, s in col_conv+row_conv:
#             for i in range(2):
#                 output_size[i] = math.floor((output_size[i] - (k[i]-1) - 1)/s[i] + 1)
#         self.output_size = output_size[0]*output_size[1]*row_conv[-1][1]

#         # self.term_linear = nn.ModuleList([nn.Linear(i, o, bias=False) for i, o in term_layers])
#         # self.term_bn = nn.ModuleList([nn.BatchNorm1d(o) for _, o in term_layers])

#         # self.linear = nn.ModuleList([nn.Linear(i, o, bias=False) for i, o in layers])
#         # self.bn = nn.ModuleList([nn.BatchNorm1d(o) for _, o in layers])

#         self.c_conv = nn.ModuleList([nn.Conv2d(i, o, k, s, bias=False) for i, o, k, s in col_conv])
#         self.c_bn = nn.ModuleList([nn.BatchNorm2d(o) for _, o, _, _ in col_conv])

#         self.r_conv = nn.ModuleList([nn.Conv2d(i, o, k, s) for i, o, k, s in row_conv])
#         self.r_bn = nn.ModuleList([nn.BatchNorm2d(o) for _, o, _, _ in row_conv])

#         self.l1 = nn.Linear(self.output_size, 128)
#         self.dropout = nn.Dropout(0.5)
#         self.l2 = nn.Linear(128, 1)

#         self.sel_l1 = nn.Linear(num_keywords, 128)
#         self.sel_l2 = nn.Linear(128, 1)

#         self.act = nn.GELU()

#         # self.last_class = nn.Linear(*class_layer)
        
#     def forward(self, x):
#         batch_size = x.shape[0]
#         norm = torch.linalg.norm(x, ord=2, dim=2)

#         x = x.unsqueeze(1)
#         for conv, bn in zip(self.c_conv, self.c_bn):
#             x = self.act(bn(conv(x)))
        
#         for conv, bn in zip(self.r_conv, self.r_bn):
#             x = self.act(bn(conv(x)))

#         x = x.view(batch_size, -1)
#         x = self.act(self.l1(x))
#         x = self.dropout(x)
#         x = self.l2(x)
        
#         norm = norm.view(batch_size, self.num_keywords)
#         norm = norm/torch.max(norm)
#         norm = self.act(self.sel_l1(norm))
#         norm = self.dropout(norm)
#         norm = self.sel_l2(norm)

#         discrimination = x+norm
        
#         return discrimination

class PositionalEncoding(nn.Module):
    def __init__(self, feature_size, keyword_size, device='cuda'):
        super().__init__()

        # Positional Encoding 초기화
        # 1. 비어있는 tensor 생성
        # (keyword_size,feature_size)
        self.P_E = torch.zeros(keyword_size, feature_size, device=device)
        # 학습되는 값이 아님으로 requires_grad 을 False로 설정
        self.P_E.requires_grad = False

        # 2. pos (0~keyword_size) 생성 (row 방향 => unsqueeze(dim=1))
        pos = torch.arange(0, keyword_size, dtype=torch.float, device=device).unsqueeze(dim=1)

        # 3. _2i (0~2i) 생성 (col 방향)
        # 2i는 step = 2 를 활용하여 i의 2배수를 만듦
        _2i = torch.arange(0, feature_size, step= 2, dtype=torch.float, device=device)

        # 4. 제안된 positional encoding 생성 
        # (i가 짝수일때 : sin, 홀수일때 : cos)
        self.P_E[:, 0::2] = torch.sin(pos / 10000 ** (_2i / feature_size))
        self.P_E[:, 1::2] = torch.cos(pos / 10000 ** (_2i / feature_size))

    def forward(self,x):
        # x seq 길이에 맞춰 PE return 
        # (seq_len, feature_size)
        seq_len = x.shape[1]
        PE_for_x = self.P_E[:seq_len,:]

        return PE_for_x
