import torch.nn as nn
import functools
import torch
import functools
import torch.nn.functional as F


# from torch.autograd import Variable


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class PATBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, cated_stream2=False):
        super(PATBlock, self).__init__()
        self.conv_block_stream1 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias,
                                                        cal_att=False)
        self.conv_block_stream2 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias,
                                                        cal_att=True, cated_stream2=cated_stream2)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, cated_stream2=False,
                         cal_att=False):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if cated_stream2:
            conv_block += [nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim * 2),
                           nn.ReLU(True)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim),
                           nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if cal_att:
            if cated_stream2:
                conv_block += [nn.Conv2d(dim * 2, dim, kernel_size=3, padding=p, bias=use_bias)]
            else:
                conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x1, x2):
        x1_out = self.conv_block_stream1(x1)
        x2_out = self.conv_block_stream2(x2)
        att = torch.sigmoid(x2_out)

        x1_out = x1_out * att
        out = x1 + x1_out  # residual connection

        # stream2 receive feedback from stream1
        x2_out = torch.cat((x2_out, out), 1)
        return out, x2_out, x1_out


class PATNModel(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', n_downsampling=2):
        assert (n_blocks >= 0 and type(input_nc) == list)
        super(PATNModel, self).__init__()
        self.input_nc_s1 = input_nc[0]
        self.input_nc_s2 = input_nc[1]
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # down_sample
        model_stream1_down = [nn.ReflectionPad2d(3),
                              nn.Conv2d(self.input_nc_s1, ngf, kernel_size=7, padding=0,
                                        bias=use_bias),
                              norm_layer(ngf),
                              nn.ReLU(True)]

        model_stream2_down = [nn.ReflectionPad2d(3),
                              nn.Conv2d(self.input_nc_s2, ngf, kernel_size=7, padding=0,
                                        bias=use_bias),
                              norm_layer(ngf),
                              nn.ReLU(True)]

        # n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model_stream1_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                             stride=2, padding=1, bias=use_bias),
                                   norm_layer(ngf * mult * 2),
                                   nn.ReLU(True)]
            model_stream2_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                             stride=2, padding=1, bias=use_bias),
                                   norm_layer(ngf * mult * 2),
                                   nn.ReLU(True)]

        # att_block in place of res_block
        mult = 2 ** n_downsampling
        cated_stream2 = [True for i in range(n_blocks)]
        cated_stream2[0] = False
        attBlock = nn.ModuleList()
        for i in range(n_blocks):
            attBlock.append(
                PATBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                         use_bias=use_bias, cated_stream2=cated_stream2[i]))

        # up_sample
        model_stream1_up = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model_stream1_up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1,
                                                    bias=use_bias),
                                 norm_layer(int(ngf * mult / 2)),
                                 nn.ReLU(True)]

        model_stream1_up += [nn.ReflectionPad2d(3)]
        model_stream1_up += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_stream1_up += [nn.Tanh()]

        # self.model = nn.Sequential(*model)
        self.stream1_down = nn.Sequential(*model_stream1_down)
        self.stream2_down = nn.Sequential(*model_stream2_down)
        # self.att = nn.Sequential(*attBlock)
        self.att = attBlock
        self.stream1_up = nn.Sequential(*model_stream1_up)

    def forward(self, input):  # x from stream 1 and stream 2
        # here x should be a tuple
        x1, x2 = input
        # down_sample
        x1 = self.stream1_down(x1)
        x2 = self.stream2_down(x2)
        # att_block
        for model in self.att:
            x1, x2, _ = model(x1, x2)

        # up_sample
        x1 = self.stream1_up(x1)

        return x1


class PATNetwork(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', n_downsampling=2):
        super(PATNetwork, self).__init__()
        assert type(input_nc) == list and len(input_nc) == 2, 'The AttModule take input_nc in format of list only!!'
        self.gpu_ids = gpu_ids
        self.model = PATNModel(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, gpu_ids, padding_type,
                               n_downsampling=n_downsampling)

    def forward(self, input):
        if self.gpu_ids and isinstance(input[0].data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


def gen_enc_fea_all_parts_cuda(rawFea, srcBox, dstBox):
    # bbox:[x1, y1, x2, y2]
    # def where(cond, x_1, x_2):
    #     cond = cond.float()    
    #     return (cond * x_1) + ((1-cond) * x_2)

    batch, channel, height, width = rawFea.size()
    maskNewFea = torch.zeros(rawFea.size()).cuda()

    srcBox = srcBox + 0.5
    srcBox = srcBox.int()

    dstBox = dstBox + 0.5
    dstBox = dstBox.int()

    for ind in range(batch):
        # process each segment seperately 
        for part in range(srcBox[ind].shape[0]):

            sx1, sy1, sx2, sy2 = srcBox[ind][part]
            dx1, dy1, dx2, dy2 = dstBox[ind][part]

            if sx1 == 0 and sy1 == 0 and sx2 == 0 and sy2 == 0:
                continue
            if dx1 == 0 and dy1 == 0 and dx2 == 0 and dy2 == 0:
                continue

            srcH, srcW = sy2 - sy1, sx2 - sx1

            maskRawFea = torch.zeros(1, channel, srcH, srcW).cuda()
            maskRawFea[:, :, :, :] = rawFea[ind, :, sy1:sy2, sx1:sx2]

            # maskRawFea = Variable(maskRawFea, requires_grad=False).cuda()

            dstH, dstW = (dy2 - dy1), (dx2 - dx1)
            if dstW <= 0 or dstH <= 0:
                continue
            grid = torch.zeros(1, dstH, dstW, 2).cuda()  # batch=1 in this case.
            # x cord.
            x_map = torch.range(1, dstW, 1) / float(dstW) * 2. - 1
            # y cord.
            y_map = torch.range(1, dstH, 1) / float(dstH) * 2. - 1

            meshgrid = torch.meshgrid(y_map, x_map)

            grid[:, :, :, 0] = meshgrid[1]
            grid[:, :, :, 1] = meshgrid[0]

            # grid = Variable(grid, requires_grad=False).cuda()

            _newFea = F.grid_sample(maskRawFea, grid, mode='nearest', padding_mode='border')

            maskNewFea[ind, :, dy1:dy2, dx1:dx2] += maskNewFea[ind, :, dy1:dy2, dx1:dx2]

    # maskNewFea = Variable(maskNewFea).cuda()
    return maskNewFea


class PATModel_Fine(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', n_downsampling=2):
        assert (n_blocks >= 0 and type(input_nc) == list)
        super(PATModel_Fine, self).__init__()
        self.input_nc_s1 = input_nc[0]
        self.input_nc_s2 = input_nc[1]
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # down_sample
        model_stream1_down = [
            nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(self.input_nc_s1, ngf, kernel_size=7, padding=0,
                          bias=use_bias),
                norm_layer(ngf),
                nn.ReLU(True)
            )]

        model_stream2_down = [nn.ReflectionPad2d(3),
                              nn.Conv2d(self.input_nc_s2, ngf, kernel_size=7, padding=0,
                                        bias=use_bias),
                              norm_layer(ngf),
                              nn.ReLU(True)]

        # n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model_stream1_down += [
                nn.Sequential(
                    nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                              stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True)
                )]
            model_stream2_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                             stride=2, padding=1, bias=use_bias),
                                   norm_layer(ngf * mult * 2),
                                   nn.ReLU(True)]

        mult = 2 ** n_downsampling
        cated_stream2 = [True for i in range(n_blocks)]
        cated_stream2[0] = False
        attBlock = nn.ModuleList()
        for i in range(n_blocks):
            attBlock.append(
                PATBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                         use_bias=use_bias, cated_stream2=cated_stream2[i]))

        # up_sample
        model_stream1_up = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model_stream1_up += [
                nn.Sequential(
                    ResnetBlock(ngf * mult * 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                                use_bias=use_bias),
                    ResnetBlock(ngf * mult * 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                                use_bias=use_bias),
                    ResnetBlock(ngf * mult * 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                                use_bias=use_bias),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult / 2),
                                       kernel_size=3, stride=2,
                                       padding=1, output_padding=1,
                                       bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True)
                )]

        model_stream1_up += [
            nn.Sequential(
                ResnetBlock(ngf * 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                            use_bias=use_bias),
                ResnetBlock(ngf * 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                            use_bias=use_bias),
                ResnetBlock(ngf * 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                            use_bias=use_bias),
                nn.ReLU(True),
                nn.Conv2d(ngf * 2, ngf, kernel_size=3,
                          stride=1, padding=1, bias=use_bias),
                norm_layer(ngf),
                nn.ReLU(True),
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                nn.Tanh()
            )]

        self.stream1_down = nn.Sequential(*model_stream1_down)
        self.stream2_down = nn.Sequential(*model_stream2_down)
        self.att = attBlock
        self.stream1_up = nn.Sequential(*model_stream1_up)

    def forward(self, input):  # x from stream 1 and stream 2
        # here x should be a tuple
        x1, x2, torso_bbox_src, torso_bbox_dst = input
        # down_sample
        # keep down_sample features for future concat.
        down_results = []  # r_7x7, r_3x3_first, r_3x3_second.
        for submodel in self.stream1_down:
            x1 = submodel(x1)
            down_results.append(x1)

        x2 = self.stream2_down(x2)
        # att_block
        for model in self.att:
            x1, x2, _ = model(x1, x2)

        enc_len = len(down_results)
        # concat and upsample.
        for i, submodel in enumerate(self.stream1_up):
            enc_fea = down_results[enc_len - 1 - i]
            if i == 0:  # 3x3
                srcBox = torch.floor_divide(torso_bbox_src, 4)
                dstBox = torch.floor_divide(torso_bbox_dst, 4)
                enc_fea = gen_enc_fea_all_parts_cuda(enc_fea, srcBox, dstBox)
            elif i == 1:  # 3x3
                srcBox = torch.floor_divide(torso_bbox_src, 2)
                dstBox = torch.floor_divide(torso_bbox_dst, 2)
                enc_fea = gen_enc_fea_all_parts_cuda(enc_fea, srcBox, dstBox)
            elif i == 2:  # 7x7
                srcBox = torso_bbox_src
                dstBox = torso_bbox_dst
                enc_fea = gen_enc_fea_all_parts_cuda(enc_fea, srcBox, dstBox)
            else:
                raise Exception('i should not be larger than 2.')

            x1 = torch.cat((x1, enc_fea), 1)
            x1 = submodel(x1)

        return x1


class PATNetwork_Fine(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', n_downsampling=2):
        super(PATNetwork_Fine, self).__init__()
        assert type(input_nc) == list and len(input_nc) == 2, 'Att_v2 take input_nc in format of list only!!'
        self.gpu_ids = gpu_ids
        self.model = PATModel_Fine(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, gpu_ids, padding_type,
                                   n_downsampling=n_downsampling)

    def forward(self, input):
        # return self.model(input)
        if self.gpu_ids and isinstance(input[0].data, torch.cuda.FloatTensor):
            # print(type(input), len(input))
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
