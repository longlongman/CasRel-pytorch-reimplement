class Config(object):
    def __init__(self, args):
        self.args = args

        # train hyper parameter
        self.multi_gpu = args.multi_gpu
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.max_len = args.max_len
        self.rel_num = args.rel_num

        # dataset
        self.dataset = args.dataset

        # path and name
        self.root = '/home/data_ti5_d/longsy/CasRel-reimplement'
        self.data_path = self.root + '/data/' + self.dataset
        self.checkpoint_dir = self.root + '/checkpoint/' + self.dataset
        self.log_dir = self.root + '/log/' + self.dataset
        self.result_dir = self.root + '/result/' + self.dataset
        self.train_prefix = args.train_prefix
        self.dev_prefix = args.dev_prefix
        self.test_prefix = args.test_prefix
        self.model_save_name = args.model_name + '_DATASET_' + self.dataset + "_LR_" + str(self.learning_rate) + "_BS_" + str(self.batch_size)
        self.log_save_name = 'LOG_' + args.model_name + '_DATASET_' + self.dataset + "_LR_" + str(self.learning_rate) + "_BS_" + str(self.batch_size)
        self.result_save_name = 'RESULT_' + args.model_name + '_DATASET_' + self.dataset + "_LR_" + str(self.learning_rate) + "_BS_" + str(self.batch_size) + ".json"

        # log setting
        self.period = args.period
        self.test_epoch = args.test_epoch

        # debug
        self.debug = args.debug
        if self.debug:
            self.dev_prefix = self.train_prefix
            self.test_prefix = self.train_prefix

