import logging
import os

def setup_logger():
    """
    AIDAN'ın tüm düşünce süreçlerini tek bir merkezden kaydeden şeffaflık motoru.
    MCTS, Lean ve Transformer modülleri birbirini bilmeden buraya rapor verir.
    """
    log_dir = "model_data"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, 'aidan_training.log')
    
    # daha once kuruluysa üzerine yazmasın. coklu cagırıda hata olmasın diey
    logger = logging.getLogger('AIDAN_Brain')
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        #dosyaya yazdır
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        #durus formatı
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        
    return logger
aidan_log = setup_logger()