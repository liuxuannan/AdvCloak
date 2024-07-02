python3 test_privacy.py --gpu 2 --batch-size 500  --query_image_dir ../data/ --gallery_image_type lfw --lenth 13233 --gallery_noise_dir ../data/lfw_112x112/ --gallery_noise_list ../data/lfw.txt --msk_dir ../generation/mask_out/ --pretrained ./target_model/r50_webface_arc_bs/model,146 --query_test_image_list ../data/privacy_ms90w_test.lst --test_img_per_id 10 

python3 test_privacy.py --gpu 2 --batch-size 500  --query_image_dir ../data/ --gallery_image_type lfw --lenth 13233 --gallery_noise_dir ../data/lfw_112x112/ --gallery_noise_list ../data/lfw.txt --msk_dir ../generation/mask_out/ --pretrained ./target_model/r50_webface_cos_bs/model,82 --query_test_image_list ../data/privacy_ms90w_test.lst --test_img_per_id 10 

python3 test_privacy.py --gpu 2 --batch-size 500  --query_image_dir ../data/ --gallery_image_type lfw --lenth 13233 --gallery_noise_dir ../data/lfw_112x112/ --gallery_noise_list ../data/lfw.txt --msk_dir ../generation/mask_out/ --pretrained ./target_model/r50_webface_sFace/model_s64a80tb87c80d12,71 --query_test_image_list ../data/privacy_ms90w_test.lst --test_img_per_id 10 

python3 test_privacy.py --gpu 2 --batch-size 500  --query_image_dir ../data/ --gallery_image_type lfw --lenth 13233 --gallery_noise_dir ../data/lfw_112x112/ --gallery_noise_list ../data/lfw.txt --msk_dir ../generation/mask_out/ --pretrained ./target_model/m1_webface_soft/model,51 --query_test_image_list ../data/privacy_ms90w_test.lst --test_img_per_id 10  

python3 test_privacy.py --gpu 2 --batch-size 500  --query_image_dir ../data/ --gallery_image_type lfw --lenth 13233 --gallery_noise_dir ../data/lfw_112x112/ --gallery_noise_list ../data/lfw.txt --msk_dir ../generation/mask_out/ --pretrained ./target_model/ser50_webface_soft/model,25 --query_test_image_list ../data/privacy_ms90w_test.lst --test_img_per_id 10 
 
python3 test_privacy.py --gpu 2 --batch-size 100  --query_image_dir ../data/ --gallery_image_type lfw --lenth 13233 --gallery_noise_dir ../data/lfw_112x112/ --gallery_noise_list ../data/lfw.txt --msk_dir ../generation/mask_out/ --pretrained ./target_model/i1_webface_soft/model_s3,21 --query_test_image_list ../data/privacy_ms90w_test.lst --test_img_per_id 10  
