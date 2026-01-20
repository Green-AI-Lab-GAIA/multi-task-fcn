#!/bin/bash

mkdir -p ./sidroforest
cd ./sidroforest

# Opções para lidar com servidores lentos/ocupados:
# --tries=10       : tenta 10 vezes em caso de erro
# --waitretry=30   : espera 30 segundos entre tentativas
# --continue       : continua downloads interrompidos
# --timeout=60     : timeout de 60 segundos
# sleep 5          : pausa entre downloads para evitar rate limiting

wget --tries=10 --waitretry=30 --continue --timeout=60 \
  https://download.pangaea.de/dataset/933263/files/Kruse_et_al_SiDroForest_RGB_Orthomosiac.zip

sleep 5

wget --tries=10 --waitretry=30 --continue --timeout=60 \
  https://download.pangaea.de/dataset/933263/files/Kruse_et_al_SiDroForest_RGN_Orthomosaic_1.zip

sleep 5

wget --tries=10 --waitretry=30 --continue --timeout=60 \
  https://download.pangaea.de/dataset/933263/files/Kruse_et_al_SiDroForest_RGN_Orthomosaic_2.zip

sleep 5

wget --tries=10 --waitretry=30 --continue --timeout=60 \
  https://download.pangaea.de/dataset/933263/files/Kruse_et_al_SiDroForest_Crowns_Polygon.zip

sleep 5

wget --tries=10 --waitretry=30 --continue --timeout=60 \
  https://download.pangaea.de/dataset/933263/files/Kruse_et_al_SiDroForest_Outer_Polygon.zip

sleep 5

wget --tries=10 --waitretry=30 --continue --timeout=60 \
  https://download.pangaea.de/dataset/933263/files/README-Kruse_et_al_SiDroForest_Orthoimages_Pointclouds.pdf

echo "Download concluído!"
