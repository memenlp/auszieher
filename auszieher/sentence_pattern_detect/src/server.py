import os

from .commons import config
from . import Model


class Server:
    def __init__(self, logger = None):
        if logger is None:
            from .commons.logger import logger
            self.logger = logger
        else:
            self.logger = logger
        # 加载模型，区分中文/英文
        data_dir = config['DATA_DIR']
        batch_size = config['BATCH_SIZE']
        self.logger = logger
        self.hypo_model = Model(os.path.join(data_dir, 'hypo'), batch_size = batch_size, logger = self.logger)

    def process(self, text_list, lang):
        self.logger.info(f'text_list:{text_list}, lang:{lang}')
        result = self.hypo_model.process(text_list, lang)
        return result


if __name__ == '__main__':
    text_list = ['it would be better if the watch screen is touch', 'dasdasd'] * 10
    lang = 'en'
    obj = Server()
    print(obj.process(text_list, lang))
