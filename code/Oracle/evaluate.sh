#!/bin/bash

insta="/users/mmazuecos/anaconda3/envs/visdial/bin/python"
avgpu="0"

lxmertbin="-config big -load_bin_path ./bin/Oracle/lxmert_big --modelname lxmert -set test"
evalpos="-config big -load_bin_path ./bin/Oracle/lxmert_big --modelname evalpos -set test --history True"
pos200="-config big -load_bin_path ./bin/Oracle/posHist_200sents_epoch15 --modelname pos200 -set test --history True"
dlxmert="-set test -load_bin_path bin/Oracle/oracledlxmerte4"
posdlxmert="-set test -load_bin_path bin/Oracle/oracleposdlxmerte26"
objposdlxmert="-set test -load_bin_path bin/Oracle/oracleobjposdlxmert49"
mixLL="-set test -load_bin_path bin/Oracle/oraclemixLL48"

printf "EVALUATION. COMPUTING CSVs\n\n\n"

printf "Evaluating LXMERT in whole test set\n\n"

CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.evaluate_lxmert_oracle_input_target $lxmertbin

printf "Evaluating EVALPOS in whole test set\n\n"

CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.evaluate_lxmert_oracle_input_target $evalpos --history True

printf "Evaluating POS200 in whole test set\n\n"

CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.evaluate_lxmert_oracle_input_target $pos200 --history True

printf "Evaluating DLXMERT in whole test set\n\n"

CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.evaluate_DLXMERTe $dlxmert

printf "Evaluating POSDLXMERT in whole test set\n\n"

CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.evaluate_POSDLXMERT $posdlxmert

printf "Evaluating OBJPOSDLXMERT in whole test set\n\n"

CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.evaluate_POSDLXMERT $objposdlxmert -onlyobj True

printf "Evaluating MIXLL in whole test set\n\n"

CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.evaluate_mixLL $mixLL -onlyobj False

printf "Computing Confusion Matrixs\n\n"

savein="./"

lxmert="lxmerttestpredictions.csv"
histlxmert="historicallxmerttestpredictions.csv"
evalpos="evalpostestpredictions.csv"
histevalpos="historicalevalpostestpredictions.csv"
pos200="pos200testpredictions.csv"
histpos200="historicalpos200testpredictions.csv"
dlxmert="dlxmerttestpredictions.csv"
histdlxmert="historicaldlxmerttestpredictions.csv"
posdlxmert="posdlxmerttestpredictions.csv"
poshistdlxmert="historicalposdlxmerttestpredictions.csv"
objposdlxmert="objposdlxmerttestpredictions.csv"
objposhistdlxmert="historicalobjposdlxmerttestpredictions.csv"
mixLL="mixLLtestpredictions.csv"
histmixLL="historicalmixLLtestpredictions.csv"

CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.getConfMatrix -data $lxmert -where $savein
CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.getConfMatrix -data $evalpos -where $savein
CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.getConfMatrix -data $pos200 -where $savein
CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.getConfMatrix -data $dlxmert -where $savein
CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.getConfMatrix -data $posdlxmert -where $savein
CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.getConfMatrix -data $objposdlxmert -where $savein
CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.getConfMatrix -data $mixLL -where $savein

printf "Confusion matrixs ready in root\n\n"

printf "Compute by category (whole test set)\n\n"

printf "LXMERT:\n"
CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.evaluate_byclass -data $lxmert -name LXMERT -is_historical False
printf "EVALPOS:\n"
CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.evaluate_byclass -data $evalpos -name LXMERT -is_historical False
printf "POS200:\n"
CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.evaluate_byclass -data $pos200 -name LXMERT -is_historical False
printf "DLXMERT:\n"
CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.evaluate_byclass -data $dlxmert -name LXMERT -is_historical False
printf "POSDLXMERT:\n"
CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.evaluate_byclass -data $posdlxmert -name LXMERT -is_historical False
printf "OBJPOSDLXMERT:\n"
CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.evaluate_byclass -data $objposdlxmert -name LXMERT -is_historical False
printf "MIXLL:\n"
CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.evaluate_byclass -data $mixLL -name LXMERT -is_historical False


printf "EVALUATING IN MINI DATASET HISTORICAL"

CUDA_VISIBLE_DEVICES=$avgpu $insta -m utils.evaluate_minitest -models lxmert,dlxmert,pos200,evalpos,posdlxmert,objposdlxmert,mixLL


