class ParamCollection():
    def __init__(self):
        self.TF_CALLBACKS = {}
        self.TORCH_CALLBACKS = {}
        self.TF_LOSSES = {}
        self.TORCH_LOSSES = {}
        self.LOSS_LOGGER_MAKERS = {}
        self.TF_MODELS = {}
        self.TORCH_MODELS = {}
        self.get_tf_optimizer_maker = None
        self.get_data = None
        self.get_model_maker = None