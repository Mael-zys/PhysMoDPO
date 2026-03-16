mkdir -p body_models
cd body_models/

echo -e "The smpl files will be stored in the 'body_models/smpl/' folder\n"
gdown "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2"
rm -rf smpl

unzip smpl.zip
echo -e "Cleaning\n"
rm smpl.zip


echo -e "The smplh files will be stored in the 'body_models/smplh/' folder\n"
gdown "https://drive.google.com/file/d/1ymu8s1svP70IJU1RV3ZPK6VjPeSzKQ3P"
rm -rf smplh

unzip smplh.zip
echo -e "Cleaning\n"
rm smplh.zip

echo -e "Downloading done!"