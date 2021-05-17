python3 preprocess.py --source-lang en --target-lang de --trainpref data-bin/iwslt14.tokenized.de-en/WMT/train --validpref data-bin/iwslt14.tokenized.de-en/WMT/valid --testpref data-bin/iwslt14.tokenized.de-en/WMT/test --destdir data-bin/iwslt14.tokenized.de-en/WMT/preprocessed --dataset-impl 'cached' --bpe 'subword_nmt' --nwordssrc 32000 --nwordstgt 32000 --joined-dictionary --workers 32
python3 joint_train.py --data data-bin/iwslt14.tokenized.de-en/WMT/preprocessed/  --src_lang en --trg_lang de --learning_rate 1e-3 --joint-batch-size 100 --gpuid 0 --clip-norm 1.0 --epochs 40
python3 generate.py --data data-bin/iwslt14.tokenized.de-en/WMT/preprocessed/  --src_lang en --trg_lang de --batch-size 64 --gpuid 0

bash postprocess.sh < real.txt > real_processed.txt
bash postprocess.sh < predictions.txt > predictions_processed.txt
perl scripts/multi-bleu.perl real_processed.txt < predictions_processed.txt

nohup python3 joint_train_sampling.py --data data-bin/iwslt14.tokenized.de-en/WMT/preprocessed/  --src_lang en --trg_lang de --learning_rate 1e-3 --joint-batch-size 100 --gpuid 0 --clip-norm 1.0 --epochs 40 > sampling.out 2>&1 &


python3 joint_train_22discriminator.py --data data-bin/iwslt14.tokenized.de-en/WMT/preprocessed/  --src_lang de --trg_lang en --learning_rate 1e-3 --joint-batch-size 100 --gpuid 0 --clip-norm 1.0 --epochs 40

########################################ZH-EN
bash wmt17_prepare.sh

python3 preprocess.py --source-lang zh --target-lang en --trainpref data-bin/WMT17/train.36000.bpe --validpref data-bin/WMT17/valid.36000.bpe --testpref data-bin/WMT17/test.36000.bpe --destdir data-bin/WMT17/preprocessed --dataset-impl 'cached' --bpe 'subword_nmt' --workers 32 --srcdict data-bin/WMT17/vocab.36000.bpe.zh --tgtdict data-bin/WMT17/vocab.36000.bpe.en
nohup python3 joint_warm_D_S.py --data data-bin/WMT17/preprocessed/  --src_lang zh --trg_lang en --learning_rate 1e-3 --joint-batch-size 100 --gpuid 0 --clip-norm 1.0 --epochs 40 > Zhenwarm.out 2>&1 &
python3 generate_professor_for.py --data data-bin/WMT17/preprocessed/  --src_lang zh --trg_lang en --batch-size 2000 --gpuid 0
python3 generate.py --data data-bin/WMT17/preprocessed/  --src_lang zh --trg_lang en --batch-size 2000 --gpuid 0


python3 preprocess.py --source-lang zh --target-lang en --trainpref data-bin/WMT17/train.36000.bpe --validpref data-bin/WMT17/pro/valid.36000.bpe --testpref data-bin/WMT17/pro/test.36000.bpe --destdir data-bin/WMT17/pro/pro_preprocessed --dataset-impl 'cached' --bpe 'subword_nmt' --workers 64 --srcdict data-bin/WMT17/pro/vocab.36000.bpe.zh --tgtdict data-bin/WMT17/pro/vocab.36000.bpe.en
nohup python3 train_generator.py --data data-bin/WMT17/pro/pro_preprocessed/  --src_lang zh --trg_lang en --learning_rate 1e-3 --batch-size 240 --gpuid 0 --clip-norm 1.0 --epochs 40 > traingenerator.out 2>&1 &
nohup python3 joint_warm_D_S.py --data data-bin/WMT17/pro/pro_preprocessed/  --src_lang zh --trg_lang en --learning_rate 1e-3 --joint-batch-size 64 --gpuid 0 --clip-norm 1.0 --epochs 40 > jointwarmup10.out 2>&1 &

python3 generate.py --data data-bin/WMT17/pro/pro_preprocessed/  --src_lang zh --trg_lang en --batch-size 2000 --gpuid 0
python3 preprocess.py --source-lang zh --target-lang en --trainpref data-bin/WMT17/pro/train --validpref data-bin/WMT17/pro/valid --testpref data-bin/WMT17/pro/test --destdir data-bin/WMT17/pro/pro_preprocessed --thresholdsrc 3 --thresholdtgt 3 --dataset-impl 'cached' --bpe 'subword_nmt' --workers 64 --srcdict data-bin/WMT17/pro/vocab.32k.cnt.zh.txt --tgtdict data-bin/WMT17/pro/vocab.32k.cnt.en.txt
nohup python3 train_generator.py --data data-bin/WMT17/pro/pro_preprocessed/  --src_lang zh --trg_lang en --learning_rate 1e-3 --batch-size 200 --gpuid 0 --clip-norm 1.0 --epochs 40 > traingenerator.out 2>&1 &
python3 generate.py --data data-bin/WMT17/pro/pro_preprocessed/  --src_lang zh --trg_lang en --batch-size 2000 --gpuid 0
python3 generate_professor.py --data data-bin/WMT17/pro/pro_preprocessed/  --src_lang zh --trg_lang en --batch-size 2000 --gpuid 0
python3 generate_professor_for.py --data data-bin/WMT17/pro/pro_preprocessed/  --src_lang zh --trg_lang en --batch-size 2001 --gpuid 0

nohup python3 train_discriminator.py --data data-bin/WMT17/pro/pro_preprocessed/  --src_lang zh --trg_lang en --learning_rate 1e-3 --batch-size 400 --gpuid 0 --clip-norm 1.0 --epochs 40 > traindiscri.out 2>&1 &
nohup python3 joint_warm_D_S.py --data data-bin/WMT17/pro/pro_preprocessed/  --src_lang zh --trg_lang en --learning_rate 1e-3 --joint-batch-size 60 --gpuid 0 --clip-norm 1.0 --epochs 40 > jointwarmup10.out 2>&1 &
nohup python3 joint_warm_D_S.py --data data-bin/WMT17/pro/pro_preprocessed/  --src_lang zh --trg_lang en --learning_rate 1e-3 --joint-batch-size 60 --gpuid 0 --clip-norm 5.0 --epochs 40 > jointwarmup10clip5.out 2>&1 &



nohup python3 joint_train_22discriminator.py --data data-bin/WMT17/pro/pro_preprocessed/  --src_lang zh --trg_lang en --learning_rate 1e-3 --joint-batch-size 100 --gpuid 0 --clip-norm 1.0 --epochs 40 > realmyzhenup5.out 2>&1 &

nohup python3 joint_train_22discriminator.py --data data-bin/WMT17/pro/pro_preprocessed/  --src_lang zh --trg_lang en --learning_rate 5e-4 --joint-batch-size 64 --gpuid 0 --clip-norm 1.0 --epochs 40 --lr_shrink 0.6 > realmyzhenup10shrink07new.out 2>&1 &

#毕设
nohup python3 joint_train.py --data data-bin/UN/million6way/fr-es/bpe/preprocessed/  --src_lang fr --trg_lang es --learning_rate 3e-4 --joint-batch-size 120 --gpuid 0 --clip-norm 1.0 --epochs 40 --save-dir data-bin/UN/million6way/fr-es/checkpoints/lr34clip1 > data-bin/UN/million6way/fr-es/checkpoints/lr34clip1.out 2>&1 &
nohup python3 joint_train.py --data data-bin/UN/million6way/en-fr/bpe/preprocessed/  --src_lang en --trg_lang fr --learning_rate 3e-4 --joint-batch-size 160 --gpuid 0 --clip-norm 1.0 --epochs 40 --save-dir data-bin/UN/million6way/en-fr/checkpoints/lr34clip1 > data-bin/UN/million6way/en-fr/checkpoints/lr34clip1.out 2>&1 &
nohup python3 joint_train.py --data data-bin/UN/million6way/en-es/bpe/preprocessed/  --src_lang en --trg_lang es --learning_rate 3e-4 --joint-batch-size 160 --gpuid 0 --clip-norm 1.0 --epochs 40 --save-dir data-bin/UN/million6way/en-es/checkpoints/lr34clip1 > data-bin/UN/million6way/en-es/checkpoints/lr34clip1.out 2>&1 &
nohup python3 joint_train.py --data data-bin/UN/million6way/en-fe/bpe/preprocessed/  --src_lang en --trg_lang fr --learning_rate 3e-4 --joint-batch-size 200 --gpuid 0 --clip-norm 1.0 --epochs 80 --save-dir data-bin/UN/million6way/en-fe/checkpoints/lr34clip1 > data-bin/UN/million6way/en-fe/checkpoints/lr34clip1.out 2>&1 &

nohup python3 joint_train.py --data data-bin/UN/million6way/enfr-fren/bpe/preprocessed/  --src_lang en --trg_lang fr --learning_rate 3e-4 --joint-batch-size 200 --gpuid 0 --clip-norm 1.0 --epochs 80 --save-dir data-bin/UN/million6way/enfr-fren/checkpoints/lr34clip1 > data-bin/UN/million6way/enfr-fren/checkpoints/lr34clip1.out 2>&1 &
nohup python3 joint_train.py --data data-bin/UN/million6way/enes-esen/bpe/preprocessed/  --src_lang en --trg_lang es --learning_rate 3e-4 --joint-batch-size 300 --gpuid 0 --clip-norm 1.0 --epochs 80 --save-dir data-bin/UN/million6way/enes-esen/checkpoints/lr34clip1 > data-bin/UN/million6way/enes-esen/checkpoints/lr34clip1.out 2>&1 &
nohup python3 joint_train.py --data data-bin/UN/million6way/enfrenes-frenesen/bpe/preprocessed/  --src_lang en --trg_lang fr --learning_rate 3e-4 --joint-batch-size 300 --gpuid 0 --clip-norm 1.0 --epochs 80 --save-dir data-bin/UN/million6way/enfrenes-frenesen/checkpoints/lr34clip1 > data-bin/UN/million6way/enfrenes-frenesen/checkpoints/lr34clip1.out 2>&1 &
