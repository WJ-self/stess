import torch
import torchvision
import torch.nn.functional as f

import math
from tqdm import tqdm

from models.segformer_encoder import SegformerEncoder
from utils import radam
import utils.viz_utils as viz_utils
from utils.loss_functions import TaskLoss, symJSDivLoss
from utils.viz_utils import plot_confusion_matrix
from models.MUDecoder import StyleEncoderE2VID, MUDecoder 
import training.base_trainer
from evaluation.metrics import MetricsSemseg

from e2vid.utils.loading_utils import load_model
from e2vid.image_reconstructor import ImageReconstructor

class ESSModel(training.base_trainer.BaseTrainer):
    def __init__(self, settings, train=True):
        self.is_training = train
        super(ESSModel, self).__init__(settings)
        self.do_val_training_epoch = False

    def init_fn(self):
        self.buildModels()
        self.createOptimizerDict()

        self.cycle_content_loss = torch.nn.L1Loss()
        self.cycle_pred_loss = symJSDivLoss()

        # Task Loss
        self.task_loss = TaskLoss(losses=self.settings.task_loss, gamma=2.0, num_classes=self.settings.semseg_num_classes,
                                  ignore_index=self.settings.semseg_ignore_label, reduction='mean')
        self.train_statistics = {}

        self.metrics_semseg_a = MetricsSemseg(self.settings.semseg_num_classes, self.settings.semseg_ignore_label,
                                              self.settings.semseg_class_names)
        if self.settings.semseg_label_val_b:
            self.metrics_semseg_b = MetricsSemseg(self.settings.semseg_num_classes, self.settings.semseg_ignore_label,
                                                  self.settings.semseg_class_names)
            self.metrics_semseg_cycle = MetricsSemseg(self.settings.semseg_num_classes, self.settings.semseg_ignore_label,
                                                      self.settings.semseg_class_names)

    def buildModels(self):
        # Front End Sensor A
        self.front_end_sensor_a = SegformerEncoder(channels=self.settings.input_channels_a)

        # Front End Sensor B
        self.front_end_sensor_b, self.e2vid_decoder = load_model(self.settings.path_to_model)
        for param in self.front_end_sensor_b.parameters():
            param.requires_grad = False
        self.front_end_sensor_b.eval()

        self.input_height = math.ceil(self.settings.img_size_b[0] / 8.0) * 8
        self.input_width = math.ceil(self.settings.img_size_b[1] / 8.0) * 8
        if self.settings.dataset_name_b == 'DDD17_events':
            self.input_height = 120
            self.input_width = 216
        self.input_height_valid = self.input_height
        self.input_width_valid = self.input_width
        self.front_end_sensor_a = StyleEncoderE2VID(self.settings.input_channels_a,
                                                    skip_connect=self.settings.skip_connect_encoder)
        self.reconstructor = ImageReconstructor(self.front_end_sensor_b, self.input_height, self.input_width,
                                                self.settings.nr_temporal_bins_b, self.settings.gpu_device,
                                                self.settings.e2vid_config)
        self.reconstructor_valid = self.reconstructor
        if self.settings.dataset_name_b == 'DDD17_events':
            self.input_height_valid = 200
            self.input_width_valid = 352
            self.reconstructor_valid = ImageReconstructor(self.front_end_sensor_b, self.input_height_valid, self.input_width_valid,
                                                    self.settings.nr_temporal_bins_b, self.settings.gpu_device,
                                                    self.settings.e2vid_config)
        

        self.models_dict = {"front_sensor_a": self.front_end_sensor_a,
                            "front_sensor_b": self.front_end_sensor_b}

        # Task Backend
        self.task_backend = MUDecoder (input_c=256, output_c=self.settings.semseg_num_classes,
                                        skip_connect=self.settings.skip_connect_task,
                                        skip_type=self.settings.skip_connect_task_type)
        self.models_dict["back_end"] = self.task_backend

    def createOptimizerDict(self):
        """Creates the dictionary containing the optimizer for the the specified subnetworks"""
        if not self.is_training:
            self.optimizers_dict = {}
            return
        front_sensor_a_params = filter(lambda p: p.requires_grad, self.front_end_sensor_a.parameters())
        optimizer_front_sensor_a = radam.RAdam(front_sensor_a_params,
                                               lr=self.settings.lr_front,
                                               weight_decay=0.,
                                               betas=(0., 0.999))
        self.optimizers_dict = {"optimizer_front_sensor_a": optimizer_front_sensor_a}

        # Task
        back_params = filter(lambda p: p.requires_grad, self.task_backend.parameters())
        optimizer_back = radam.RAdam(back_params,
                                     lr=self.settings.lr_back,
                                     weight_decay=0.,
                                     betas=(0., 0.999))
        self.optimizers_dict["optimizer_back"] = optimizer_back

    def train_step(self, input_batch):
        """
        ESS模型的训练步骤。

        参数:
            input_batch (tuple): 输入批次，input_batch[0][x]: 图像域，input_batch[1][x]: 事件域，
            x: 0: 数据，1: 标签，2: 配对标签（如果required_paired_data_train_x为True）

        返回:
            losses (dict): 损失字典。
            outputs (dict): 输出字典。
            final_loss (float): 最终损失。
        """
        final_loss = 0.  # 初始化最终损失
        losses = {}  # 初始化损失字典
        outputs = {}  # 初始化输出字典

        # 任务步骤
        optimizers_list = ['optimizer_back']  # 初始化优化器列表
        optimizers_list.append('optimizer_front_sensor_a')  # 添加前端传感器A的优化器

        # 为每个优化器重置梯度
        for key_word in optimizers_list:
            optimizer_key_word = self.optimizers_dict[key_word]
            optimizer_key_word.zero_grad()

        # 在图像上进行训练
        t_final_loss, t_losses, t_outputs = self.img_train_step(input_batch)
        # 如果数据集名称为DDD17_events
        if self.settings.dataset_name_b == 'DDD17_events':
            t_final_loss.backward()  # 反向传播
        # 如果数据集名称为DSEC_events
        elif self.settings.dataset_name_b == 'DSEC_events':
            # 禁用前端传感器A的梯度
            for p in self.models_dict['front_sensor_a'].parameters():
                p.requires_grad = False
            t_final_loss.backward()  # 反向传播
            # 启用前端传感器A的梯度
            for p in self.models_dict['front_sensor_a'].parameters():
                p.requires_grad = True

        final_loss += t_final_loss  # 累加图像域的最终损失
        losses.update(t_losses)  # 更新损失字典
        outputs.update(t_outputs)  # 更新输出字典

        # 在事件上进行训练
        e_event_final_loss, t_event_final_loss, event_losses, event_outputs = self.event_train_step(input_batch)
        # 禁用后端模型的梯度
        for p in self.models_dict['back_end'].parameters():
            p.requires_grad = False
        e_event_final_loss.backward()  # 反向传播事件域的损失
        # 启用后端模型的梯度
        for p in self.models_dict['back_end'].parameters():
            p.requires_grad = True
        t_event_final_loss.backward()  # 反向传播任务域的损失
        final_loss += e_event_final_loss  # 累加事件域的最终损失
        final_loss += t_event_final_loss  # 累加任务域的最终损失
        losses.update(event_losses)  # 更新损失字典
        outputs.update(event_outputs)  # 更新输出字典

        # 对每个优化器进行一步优化
        for key_word in optimizers_list:
            optimizer_key_word = self.optimizers_dict[key_word]
            optimizer_key_word.step()

        return losses, outputs, final_loss  # 返回损失字典、输出字典和最终损失


    def img_train_step(self, batch):
        """
        在图像域上进行训练。

        参数:
            batch (tuple): 包含数据和标签的批次。

        返回:
            t (float): 任务损失。
            losses (dict): 损失字典。
            out (dict): 输出字典。
        """
        data_a = batch[0][0]  # 获取图像域数据

        # 根据设置决定是否需要成对数据
        if self.settings.require_paired_data_train_a:
            labels_a = batch[0][2]  # 获取成对的标签
        else:
            labels_a = batch[0][1]  # 获取非成对的标签

        # 设置BatchNorm统计数据
        for model in self.models_dict:
            self.models_dict[model].train()  # 设置为训练模式
            # 对特定模型设置为评估模式
            if model in ['front_sensor_b', 'e2vid_decoder']:
                self.models_dict[model].eval()
        # 设置后端模型参数的梯度
        for p in self.models_dict['back_end'].parameters():
            p.requires_grad = True

        gen_model_sensor_a = self.models_dict['front_sensor_a']  # 获取前端传感器A的生成模型

        losses = {}  # 初始化损失字典
        out = {}  # 初始化输出字典
        t = 0.  # 初始化任务损失

        latent_fake = gen_model_sensor_a(data_a)  # 生成伪造的潜在特征

        # 训练任务步骤
        t_loss, pred_a = self.trainTaskStep('sensor_a', latent_fake, labels_a, losses)
        t += t_loss  # 累加任务损失

        # 如果是可视化的周期，则进行可视化步骤
        if self.visualize_epoch():
            self.visTaskStep(data_a, pred_a, labels_a)

        return t, losses, out  # 返回任务损失、损失字典和输出字典


    def event_train_step(self, batch):
        """
        在事件域上进行训练。

        参数:
            batch (tuple): 包含数据和标签的批次。

        返回:
            e_loss (float): 事件域损失。
            t_loss (float): 任务损失。
            losses (dict): 损失字典。
            out (dict): 输出字典。
        """
        data_b = batch[1][0]  # 获取事件域数据
        # 根据设置决定是否需要成对数据
        if self.settings.require_paired_data_train_b:
            labels_b = batch[1][2]  # 获取成对的标签
        else:
            labels_b = batch[1][1]  # 获取非成对的标签

        # 设置BatchNorm统计数据
        for model in self.models_dict:
            self.models_dict[model].train()  # 设置为训练模式
            # 对特定模型设置为评估模式
            if model in ['front_sensor_b', 'e2vid_decoder', "back_end"]:
                self.models_dict[model].eval()

        gen_model_sensor_a = self.models_dict['front_sensor_a']  # 获取前端传感器A的生成模型
        self.reconstructor.last_states_for_each_channel = {'grayscale': None}  # 初始化重建器的状态
        e_loss = 0.  # 初始化事件域损失
        losses = {}  # 初始化损失字典
        out = {}  # 初始化输出字典

        # 训练图像编码器
        with torch.no_grad():  # 不计算梯度
            for i in range(self.settings.nr_events_data_b):
                # 获取事件张量
                event_tensor = data_b[:, i * self.settings.input_channels_b:(i + 1) * self.settings.input_channels_b, :, :]
                # 更新重建
                img_fake, states_real, latent_real = self.reconstructor.update_reconstruction(event_tensor)

        latent_fake = gen_model_sensor_a(img_fake.detach())  # 生成伪造的潜在特征

        # 分离真实的潜在特征
        for key in latent_real.keys():
            latent_real[key] = latent_real[key].detach()

        # 训练循环步骤
        cycle_loss, pred_b, pred_a = self.trainCycleStep('sensor_b', 'sensor_a', latent_real, latent_fake, losses)
        e_loss += cycle_loss  # 累加循环损失到事件域损失

        # 如果是可视化的周期，则进行可视化步骤
        if self.visualize_epoch():
            self.visCycleStep(data_b, img_fake, pred_b, pred_a, labels_b)

        # 训练任务网络
        self.models_dict['back_end'].train()  # 设置任务网络为训练模式
        t_loss = 0.  # 初始化任务损失
        # 计算任务循环步骤的损失
        t_loss += self.TasktrainCycleStep('sensor_b', 'sensor_a', latent_real, latent_fake, losses)
        # 如果设置为在事件标签上训练
        if self.settings.train_on_event_labels:
            t_loss_b, _ = self.trainTaskStep('sensor_b', latent_real, labels_b, losses)
            t_loss += t_loss_b  # 累加任务损失

        return e_loss, t_loss, losses, out  # 返回损失和输出

    
    def trainTaskStep(self, sensor_name, latent_fake, labels, losses):
        """
        计算任务损失以及任务网络的预测（语义分割）。
        在图像域中：L_{task}：任务损失。
        在事件域中：L_{cons.pred}：预测的一致性损失。

        参数:
            sensor_name (str): 图像域或事件域。
            latent_fake (Tensor): 伪造的潜在特征。
            labels (Tensor): 标签。
            losses (dict): 损失字典。

        返回:
            loss_pred (float): 任务损失。
            pred (Tensor): 预测。
        """
        content_features = {}  # 初始化内容特征字典
        # 遍历伪造的潜在特征键值
        for key in latent_fake.keys():
            # 如果数据集名称为DDD17_events
            if self.settings.dataset_name_b == 'DDD17_events':
                content_features[key] = latent_fake[key]  # 直接使用潜在特征
            # 如果数据集名称为DSEC_events
            elif self.settings.dataset_name_b == 'DSEC_events':
                content_features[key] = latent_fake[key].detach()  # 分离潜在特征
        task_backend = self.models_dict["back_end"]  # 获取模型的后端部分
        pred = task_backend(content_features)  # 使用后端模型进行预测
        # 计算预测的任务损失，并乘以任务损失权重
        loss_pred = self.task_loss(pred[1], labels) * self.settings.weight_task_loss
        losses['semseg_' + sensor_name + '_loss'] = loss_pred.detach()  # 记录损失

        return loss_pred, pred  # 返回任务损失和预测

    def trainCycleStep(self, first_sensor_name, second_sensor_name, content_first_sensor, content_second_sensor, losses):
        """
        计算两个传感器输出的嵌入向量之间的一致性损失。

        参数:
            first_sensor_name (str): 第一个传感器的名称。
            second_sensor_name (str): 第二个传感器的名称。
            content_first_sensor (Tensor): 第一个传感器的内容。
            content_second_sensor (Tensor): 第二个传感器的内容。
            losses (dict): 存储损失值的字典。

        返回:
            g_loss (float): 总的生成器损失。
            pred_first_sensor_no_grad (Tensor): 不计算梯度的第一个传感器的预测。
            pred_second_sensor (Tensor): 第二个传感器的预测。
        """
        g_loss = 0.  # 初始化生成器损失为0
        cycle_name = first_sensor_name + '_to_' + second_sensor_name  # 创建循环名称

        # 如果设置了跳过连接编码器，则计算2倍和4倍的潜在特征损失
        if self.settings.skip_connect_encoder:
            # 计算2倍潜在特征的循环内容损失，并乘以循环损失权重
            cycle_latent_loss_2x = self.cycle_content_loss(content_second_sensor[2], content_first_sensor[2]) * \
                                self.settings.weight_cycle_loss
            g_loss += cycle_latent_loss_2x  # 累加到总损失
            losses['cycle_latent_2x_' + cycle_name + '_loss'] = cycle_latent_loss_2x.cpu().detach()  # 记录损失

            # 计算4倍潜在特征的循环内容损失，并乘以循环损失权重
            cycle_latent_loss_4x = self.cycle_content_loss(content_second_sensor[4], content_first_sensor[4]) * \
                                self.settings.weight_cycle_loss
            g_loss += cycle_latent_loss_4x  # 累加到总损失
            losses['cycle_latent_4x_' + cycle_name + '_loss'] = cycle_latent_loss_4x.cpu().detach()  # 记录损失

        # 计算8倍潜在特征的循环内容损失，并乘以循环损失权重
        cycle_latent_loss_8x = self.cycle_content_loss(content_second_sensor[8], content_first_sensor[8]) * \
                            self.settings.weight_cycle_loss
        g_loss += cycle_latent_loss_8x  # 累加到总损失
        losses['cycle_latent_8x_' + cycle_name + '_loss'] = cycle_latent_loss_8x.cpu().detach()  # 记录损失

        task_backend = self.models_dict["back_end"]  # 获取模型的后端部分

        # 使用后端模型对第二个传感器的内容进行预测
        pred_second_sensor = task_backend(content_second_sensor)
        with torch.no_grad():  # 不计算梯度
            # 使用后端模型对第一个传感器的内容进行预测（不计算梯度）
            pred_first_sensor_no_grad = task_backend(content_first_sensor)

        # 计算1倍预测的循环预测损失
        cycle_pred_loss_1x_events = self.cycle_pred_loss(pred_second_sensor[1], pred_first_sensor_no_grad[1])
        losses['cycle_pred_1x_' + cycle_name + '_loss'] = cycle_pred_loss_1x_events.cpu().detach()  # 记录损失
        cycle_pred_loss_1x = cycle_pred_loss_1x_events
        if self.settings.dataset_name_b == 'DSEC_events':  # 如果数据集名称为DSEC_events
            g_loss += cycle_pred_loss_1x  # 累加到总损失

        # 计算2倍预测的循环内容损失，并乘以循环任务损失权重
        cycle_pred_loss_2x_events = self.cycle_content_loss(pred_second_sensor[2], pred_first_sensor_no_grad[2]) * \
                                    self.settings.weight_cycle_task_loss
        cycle_pred_loss_2x = cycle_pred_loss_2x_events
        g_loss += cycle_pred_loss_2x  # 累加到总损失
        losses['cycle_pred_2x_' + cycle_name + '_loss'] = cycle_pred_loss_2x.cpu().detach()  # 记录损失

        # 计算4倍预测的循环内容损失，并乘以循环任务损失权重
        cycle_pred_loss_4x_events = self.cycle_content_loss(pred_second_sensor[4], pred_first_sensor_no_grad[4]) * \
                                    self.settings.weight_cycle_task_loss
        cycle_pred_loss_4x = cycle_pred_loss_4x_events
        g_loss += cycle_pred_loss_4x  # 累加到总损失
        losses['cycle_pred_4x_' + cycle_name + '_loss'] = cycle_pred_loss_4x.cpu().detach()  # 记录损失

        return g_loss, pred_first_sensor_no_grad, pred_second_sensor  # 返回总损失和两个传感器的预测



    def TasktrainCycleStep(self, first_sensor_name, second_sensor_name, content_first_sensor, content_second_sensor, losses):
        """
        L_{cons.task}: 在事件域中的任务网络上计算一致性损失。

        参数:
            first_sensor_name (str): 第一个传感器的名称。
            second_sensor_name (str): 第二个传感器的名称。
            content_first_sensor (Tensor): 第一个传感器的内容。
            content_second_sensor (Tensor): 第二个传感器的内容。
            losses (dict): 存储损失值的字典。

        返回:
            t_loss (float): 总的任务损失。
        """
        t_loss = 0.  # 初始化任务损失为0
        cycle_name = first_sensor_name + '_to_' + second_sensor_name  # 创建循环名称

        task_backend = self.models_dict["back_end"]  # 获取模型的后端部分
        pred_first_sensor = task_backend(content_first_sensor)  # 使用后端模型对第一个传感器的内容进行预测

        with torch.no_grad():  # 不计算梯度
            pred_second_sensor_no_grad = task_backend(content_second_sensor)  # 使用后端模型对第二个传感器的内容进行预测（不计算梯度）

        # 计算1倍预测的循环预测损失，并乘以KL损失权重
        cycle_pred_loss_1x_events = self.cycle_pred_loss(pred_first_sensor[1], pred_second_sensor_no_grad[1]) * \
                                    self.settings.weight_KL_loss
        cycle_pred_loss_1x = cycle_pred_loss_1x_events
        t_loss += cycle_pred_loss_1x  # 累加到总损失

        # 计算2倍预测的循环内容损失，并乘以循环任务损失权重
        cycle_pred_loss_2x_events = self.cycle_content_loss(pred_first_sensor[2], pred_second_sensor_no_grad[2]) * \
                                    self.settings.weight_cycle_task_loss
        cycle_pred_loss_2x = cycle_pred_loss_2x_events
        t_loss += cycle_pred_loss_2x  # 累加到总损失

        # 计算4倍预测的循环内容损失，并乘以循环任务损失权重
        cycle_pred_loss_4x_events = self.cycle_content_loss(pred_first_sensor[4], pred_second_sensor_no_grad[4]) * \
                                    self.settings.weight_cycle_task_loss
        cycle_pred_loss_4x = cycle_pred_loss_4x_events
        t_loss += cycle_pred_loss_4x  # 累加到总损失

        return t_loss  # 返回总任务损失


    def visCycleStep(self, events_real, img_fake, pred_events, pred_img, labels):
        pred_events = pred_events[1]
        pred_img = pred_img[1]
        pred_events_lbl = pred_events.argmax(dim=1)
        pred_img_lbl = pred_img.argmax(dim=1)

        semseg_events = viz_utils.prepare_semseg(pred_events_lbl, self.settings.semseg_color_map,
                                                 self.settings.semseg_ignore_label)
        semseg_img = viz_utils.prepare_semseg(pred_img_lbl, self.settings.semseg_color_map,
                                              self.settings.semseg_ignore_label)
        if self.settings.semseg_label_train_b:
            semseg_gt = viz_utils.prepare_semseg(labels, self.settings.semseg_color_map, self.settings.semseg_ignore_label)

            nrow = 4
            viz_tensors = torch.cat((viz_utils.createRGBImage(events_real[:nrow],
                                                              separate_pol=self.settings.separate_pol_b).clamp(0, 1).to(
                self.device),
                                     viz_utils.createRGBImage(img_fake[:nrow]),
                                     viz_utils.createRGBImage(semseg_events[:nrow].to(self.device)),
                                     viz_utils.createRGBImage(semseg_img[:nrow].to(self.device)),
                                     viz_utils.createRGBImage(semseg_gt[:nrow].to(self.device))), dim=0)
        else:
            nrow = 4
            viz_tensors = torch.cat((viz_utils.createRGBImage(events_real[:nrow],
                                                              separate_pol=self.settings.separate_pol_b).clamp(0, 1).to(
                self.device),
                                     viz_utils.createRGBImage(img_fake[:nrow]),
                                     viz_utils.createRGBImage(semseg_events[:nrow].to(self.device)),
                                     viz_utils.createRGBImage(semseg_img[:nrow].to(self.device))), dim=0)
        rgb_grid = torchvision.utils.make_grid(viz_tensors, nrow=nrow)
        self.img_summaries('train/semseg_cycle', rgb_grid, self.step_count)
        
    def visTaskStep(self, img, pred_img, labels):
        pred_img = pred_img[1]
        pred_img_lbl = pred_img.argmax(dim=1)

        semseg_img = viz_utils.prepare_semseg(pred_img_lbl, self.settings.semseg_color_map,
                                              self.settings.semseg_ignore_label)
        semseg_gt = viz_utils.prepare_semseg(labels, self.settings.semseg_color_map, self.settings.semseg_ignore_label)

        nrow = 4
        viz_tensors = torch.cat((viz_utils.createRGBImage(img[:nrow]),
                                 viz_utils.createRGBImage(semseg_img[:nrow].to(self.device)),
                                 viz_utils.createRGBImage(semseg_gt[:nrow].to(self.device))), dim=0)
        rgb_grid = torchvision.utils.make_grid(viz_tensors, nrow=nrow)
        self.img_summaries('train/semseg_img', rgb_grid, self.step_count)

#Validation########################################################################################################################

    def validationEpoch(self, data_loader, sensor_name):
        val_dataset_length = data_loader.__len__()
        self.pbar = tqdm(total=val_dataset_length, unit='Batch', unit_scale=True)
        tqdm.write("Validation on " + sensor_name)
        cumulative_losses = {}
        total_nr_steps = None

        for i_batch, sample_batched in enumerate(data_loader):
            for i in range(len(sample_batched)):
                sample_batched[i] = sample_batched[i].to(self.device)
            self.validationBatchStep(sample_batched, sensor_name, i_batch, cumulative_losses, val_dataset_length)
            self.pbar.update(1)
            total_nr_steps = i_batch

        if sensor_name == 'sensor_a':
            metrics_semseg_a = self.metrics_semseg_a.get_metrics_summary()
            metric_semseg_a_mean_iou = metrics_semseg_a['mean_iou']
            cumulative_losses['semseg_sensor_a_mean_iou'] = metric_semseg_a_mean_iou
            metric_semseg_a_acc = metrics_semseg_a['acc']
            cumulative_losses['semseg_sensor_a_acc'] = metric_semseg_a_acc
            metrics_semseg_a_cm = metrics_semseg_a['cm']
            figure_semseg_a_cm = plot_confusion_matrix(metrics_semseg_a_cm, classes=self.settings.semseg_class_names,
                                                       normalize=True,
                                                       title='Normalized confusion matrix')
            self.summary_writer.add_figure('val_gray/semseg_cm',
                                           figure_semseg_a_cm, self.epoch_count)
        else:
            if self.settings.semseg_label_val_b:
                metrics_semseg_b = self.metrics_semseg_b.get_metrics_summary()
                metric_semseg_b_mean_iou = metrics_semseg_b['mean_iou']
                cumulative_losses['semseg_sensor_b_mean_iou'] = metric_semseg_b_mean_iou
                metric_semseg_b_acc = metrics_semseg_b['acc']
                cumulative_losses['semseg_sensor_b_acc'] = metric_semseg_b_acc
                metrics_semseg_b_cm = metrics_semseg_b['cm']
                figure_semseg_b_cm = plot_confusion_matrix(metrics_semseg_b_cm, classes=self.settings.semseg_class_names,
                                                           normalize=True,
                                                           title='Normalized confusion matrix')
                self.summary_writer.add_figure('val_events/semseg_cm',
                                               figure_semseg_b_cm, self.epoch_count)

                metrics_semseg_cycle = self.metrics_semseg_cycle.get_metrics_summary()
                metric_semseg_cycle_mean_iou = metrics_semseg_cycle['mean_iou']
                cumulative_losses['semseg_sensor_cycle_mean_iou'] = metric_semseg_cycle_mean_iou
                metric_semseg_cycle_acc = metrics_semseg_cycle['acc']
                cumulative_losses['semseg_sensor_cycle_acc'] = metric_semseg_cycle_acc
                metrics_semseg_cycle_cm = metrics_semseg_cycle['cm']
                figure_semseg_cycle_cm = plot_confusion_matrix(metrics_semseg_cycle_cm,
                                                               classes=self.settings.semseg_class_names,
                                                               normalize=True,
                                                               title='Normalized confusion matrix')
                self.summary_writer.add_figure('val_events/cycle_semseg_cm',
                                               figure_semseg_cycle_cm, self.epoch_count)

        self.val_summaries(cumulative_losses, total_nr_steps + 1)
        self.pbar.close()
        if self.val_confusion_matrix.sum() != 0:
            self.addValidationMatrix(sensor_name)

        self.saveValStatistics('val', sensor_name)

    def val_step(self, input_batch, sensor, i_batch, vis_reconstr_idx):
        """Calculates the performance measurements based on the input"""
        data = input_batch[0]
        paired_data = None
        if sensor == 'sensor_a':
            if self.settings.require_paired_data_val_a:
                paired_data = input_batch[1]
                labels = input_batch[2]
            else:
                labels = input_batch[1]
        else:
            if self.settings.require_paired_data_val_b:
                paired_data = input_batch[1]
                if self.settings.dataset_name_b == 'DDD17_events':
                    labels = input_batch[3]
                else:
                    labels = input_batch[2]
            else:
                labels = input_batch[1]

        gen_model = self.models_dict['front_' + sensor]

        losses = {}
        second_sensor = 'sensor_b'

        if sensor == 'sensor_a':
            content_first_sensor = gen_model(data)

        else:
            second_sensor = 'sensor_a'
            self.reconstructor_valid.last_states_for_each_channel = {'grayscale': None}
            for i in range(self.settings.nr_events_data_b):
                event_tensor = data[:, i * self.settings.input_channels_b:(i + 1) * self.settings.input_channels_b, :,
                               :]
                img_fake, _, content_first_sensor = self.reconstructor_valid.update_reconstruction(event_tensor)

        preds_first_sensor = self.valTaskStep(content_first_sensor, labels, losses, sensor)

        if sensor == 'sensor_b':
            preds_second_sensor = self.valCycleStep(content_first_sensor, img_fake, labels, losses, sensor,
                                                    second_sensor, preds_first_sensor)

        if vis_reconstr_idx != -1:
            if sensor == 'sensor_a':
                self.visualizeSensorA(data, labels, preds_first_sensor, vis_reconstr_idx, sensor)
            else:
                self.visualizeSensorB(data[:, -self.settings.input_channels_b:, :, :], preds_first_sensor,
                                              preds_second_sensor,
                                              labels, img_fake, paired_data,
                                              vis_reconstr_idx, sensor)
        return losses, None

    def valTaskStep(self, content_first_sensor, labels, losses, sensor):
        """Computes the task loss and updates metrics"""
        task_backend = self.models_dict["back_end"]
        preds = task_backend(content_first_sensor)

        if sensor == 'sensor_a' or self.settings.semseg_label_val_b:
            pred = preds[1]
            if sensor == 'sensor_b':
                pred = f.interpolate(pred, size=(self.settings.img_size_b), mode='nearest')
            pred_lbl = pred.argmax(dim=1)

            loss_pred = self.task_loss(pred, target=labels) * self.settings.weight_task_loss
            losses['semseg_' + sensor + '_loss'] = loss_pred.detach()
            if sensor == 'sensor_a':
                self.metrics_semseg_a.update_batch(pred_lbl, labels)
            else:
                self.metrics_semseg_b.update_batch(pred_lbl, labels)
        return preds

    def valCycleStep(self, content_first_sensor, img_fake, labels, losses, sensor, second_sensor,
                     preds_first_sensor):
        """Computes the cycle loss"""
        gen_second_sensor_model = self.models_dict['front_' + second_sensor]
        content_second_sensor = gen_second_sensor_model(img_fake)

        # latent_feature
        cycle_name = sensor + '_to_' + second_sensor
        if self.settings.skip_connect_encoder:
            cycle_latent_loss_2x = self.cycle_content_loss(content_first_sensor[2], content_second_sensor[2]) * \
                                   self.settings.weight_cycle_loss
            losses['cycle_latent_2x_' + cycle_name + '_loss'] = cycle_latent_loss_2x.cpu().detach()
            cycle_latent_loss_4x = self.cycle_content_loss(content_first_sensor[4], content_second_sensor[4]) * \
                                   self.settings.weight_cycle_loss
            losses['cycle_latent_4x_' + cycle_name + '_loss'] = cycle_latent_loss_4x.cpu().detach()
        cycle_latent_loss_8x = self.cycle_content_loss(content_first_sensor[8], content_second_sensor[8]) * \
                               self.settings.weight_cycle_loss
        losses['cycle_latent_8x_' + cycle_name + '_loss'] = cycle_latent_loss_8x.cpu().detach()

        preds_second_sensor = self.valCycleTask(content_second_sensor, labels, losses, cycle_name,
                                                    preds_first_sensor)

        return preds_second_sensor

    def valCycleTask(self, cycle_content_first_second, labels, losses, cycle_name, preds_first_sensor):
        """Computes the task performance of the E2VID reconstruction"""
        task_backend = self.models_dict["back_end"]
        preds_second_sensor = task_backend(cycle_content_first_second)
        if self.settings.semseg_label_val_b:
            pred_second_sensor = preds_second_sensor[1]
            pred_second_sensor = f.interpolate(pred_second_sensor, size=(self.settings.img_size_b), mode='nearest')
            pred_second_sensor_lbl = pred_second_sensor.argmax(dim=1)
            loss_pred = self.task_loss(pred_second_sensor, target=labels) * self.settings.weight_task_loss

            losses['semseg_' + cycle_name + '_loss'] = loss_pred.detach()
            self.metrics_semseg_cycle.update_batch(pred_second_sensor_lbl, labels)

        cycle_pred_loss_1x = self.cycle_pred_loss(preds_second_sensor[1], preds_first_sensor[1]) * \
                             self.settings.weight_KL_loss
        losses['cycle_pred_1x_' + cycle_name + '_loss'] = cycle_pred_loss_1x.cpu().detach()

        cycle_pred_loss_2x = self.cycle_content_loss(preds_first_sensor[2], preds_second_sensor[2]) * \
                             self.settings.weight_cycle_task_loss
        losses['cycle_pred_2x_' + cycle_name + '_loss'] = cycle_pred_loss_2x.cpu().detach()

        cycle_pred_loss_4x = self.cycle_content_loss(preds_first_sensor[4], preds_second_sensor[4]) * \
                             self.settings.weight_cycle_task_loss
        losses['cycle_pred_4x_' + cycle_name + '_loss'] = cycle_pred_loss_4x.cpu().detach()

        return preds_second_sensor

    def visualizeSensorA(self, data, labels, preds_first_sensor, vis_reconstr_idx, sensor):
        nrow = 4
        vis_tensors = [viz_utils.createRGBImage(data[:nrow])]

        pred = preds_first_sensor
        pred = pred[1]
        pred_lbl = pred.argmax(dim=1)

        semseg = viz_utils.prepare_semseg(pred_lbl, self.settings.semseg_color_map, self.settings.semseg_ignore_label)
        vis_tensors.append(viz_utils.createRGBImage(semseg[:nrow].to(self.device)))
        semseg_gt = viz_utils.prepare_semseg(labels, self.settings.semseg_color_map, self.settings.semseg_ignore_label)
        vis_tensors.append(viz_utils.createRGBImage(semseg_gt[:nrow]).to(self.device))

        rgb_grid = torchvision.utils.make_grid(torch.cat(vis_tensors, dim=0), nrow=nrow)
        self.img_summaries('val_' + sensor + '/reconst_input_' + sensor + '_' + str(vis_reconstr_idx),
                           rgb_grid, self.epoch_count)

    def visualizeSensorB(self, data, preds_first_sensor, preds_second_sensor, labels, img_fake, paired_data,
                                 vis_reconstr_idx, sensor):
        pred = preds_first_sensor[1]
        pred_lbl = pred.argmax(dim=1)
        pred_cycle = preds_second_sensor[1]
        pred_cycle_lbl = pred_cycle.argmax(dim=1)

        if self.settings.semseg_label_val_b:
            labels = f.interpolate(labels.float().unsqueeze(1), size=(self.input_height_valid, self.input_width_valid),
                                   mode='nearest').squeeze(1).long()

        semseg = viz_utils.prepare_semseg(pred_lbl, self.settings.semseg_color_map, self.settings.semseg_ignore_label)
        semseg_cycle = viz_utils.prepare_semseg(pred_cycle_lbl, self.settings.semseg_color_map,
                                                self.settings.semseg_ignore_label)
        if self.settings.semseg_label_val_b:
            semseg_gt = viz_utils.prepare_semseg(labels, self.settings.semseg_color_map, self.settings.semseg_ignore_label)

            nrow = 4
            vis_tensors = [
                viz_utils.createRGBImage(data[:nrow], separate_pol=self.settings.separate_pol_b).clamp(0, 1).to(
                    self.device),
                viz_utils.createRGBImage(img_fake[:nrow]).to(self.device),
                viz_utils.createRGBImage(semseg[:nrow].to(self.device)),
                viz_utils.createRGBImage(semseg_cycle[:nrow]).to(self.device),
                viz_utils.createRGBImage(semseg_gt[:nrow]).to(self.device)]

        else:
            nrow = 4
            vis_tensors = [
                viz_utils.createRGBImage(data[:nrow], separate_pol=self.settings.separate_pol_b).clamp(0, 1).to(
                    self.device),
                viz_utils.createRGBImage(img_fake[:nrow]).to(self.device),
                viz_utils.createRGBImage(semseg[:nrow].to(self.device)),
                viz_utils.createRGBImage(semseg_cycle[:nrow]).to(self.device)]

        if paired_data is not None:
            vis_tensors.append(viz_utils.createRGBImage(paired_data[:nrow]).to(self.device))

        rgb_grid = torchvision.utils.make_grid(torch.cat(vis_tensors, dim=0), nrow=nrow)
        self.img_summaries('val_' + sensor + '/reconst_input_' + sensor + '_' + str(vis_reconstr_idx),
                           rgb_grid, self.epoch_count)

    def resetValidationStatistics(self):
        self.metrics_semseg_a.reset()
        if self.settings.semseg_label_val_b:
            self.metrics_semseg_b.reset()
            self.metrics_semseg_cycle.reset()
