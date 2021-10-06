#!/bin/sh

TRAIN=/Users/xhe/Documents/NLP/Homework1/resources/data/train.jsonl
TEST=/Users/xhe/Documents/NLP/Homework1/resources/data/val.jsonl
USER_DATA=/Users/xhe/Documents/NLP/Homework1/resources/data/users.json
# {Ngram,Ngram+Lex,Ngram+Lex+Ling,Ngram+Lex+Ling+User} 
MODEL1=Ngram
MODEL2=Ngram+Lex
MODEL3=Ngram+Lex+Ling
MODEL4=Ngram+Lex+Ling+User
LEXICON_PATH=/Users/xhe/Documents/NLP/Homework1/resources/lexica/
OUTFILE=/Users/xhe/Documents/NLP/Homework1/submission/output.txt

# python hw1.py  --train $TRAIN \
#                --test $TEST \
#                --user_data $USER_DATA \
#                --model $MODEL1  \
#                --lexicon_path $LEXICON_PATH \
#                --outfile $OUTFILE
# python hw1.py  --train $TRAIN \
#                --test $TEST \
#                --user_data $USER_DATA \
#                --model $MODEL2\
#                --lexicon_path $LEXICON_PATH \
#                --outfile $OUTFILE
# python hw1.py  --train $TRAIN \
#                --test $TEST \
#                --user_data $USER_DATA \
#                --model $MODEL3\
#                --lexicon_path $LEXICON_PATH \
#                --outfile $OUTFILE
python hw1.py  --train $TRAIN \
               --test $TEST \
               --user_data $USER_DATA \
               --model $MODEL4\
               --lexicon_path $LEXICON_PATH \
               --outfile $OUTFILE