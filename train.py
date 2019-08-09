from lib.solver import CaptioningSolver
from lib.SentiAttend import CaptionGenerator
from lib.utils import load_coco_data
import os

def main():

    save_path = './model/'

    data_save_path = './data/'

    image_save_path = './images/'

    log_save_path = './log/'

    data = load_coco_data(data_path=data_save_path, split='train')
    word_to_idx = data['word_to_idx']

    val_data = load_coco_data( data_path=data_save_path, split='val' )

    test_data = load_coco_data( data_path=data_save_path, split='test' )

    model = CaptionGenerator( word_to_idx, dim_feature=[49, 2048], dim_embed=512, dim_senti=256,
                                       dim_hidden=2048, n_time_step=20, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True )

    solver = CaptioningSolver( model, data, val_data, n_epochs=30, batch_size=180, update_rule='adam',
                                          learning_rate=0.001, print_every=1000, save_every=1, image_path=image_save_path,
                                    pretrained_model=None, model_path=save_path, test_model=os.path.join(save_path+'model-SA'),
                                     print_bleu=True, log_path=log_save_path, data_save_path=data_save_path )

    solver.train()

if __name__ == "__main__":
    main()