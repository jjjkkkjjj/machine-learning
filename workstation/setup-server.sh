echo "setup ssh"

cd
mkdir .ssh
mv id_rsa.pub .ssh/
chmod 700 .ssh
cd .ssh
cat id_rsa.pub >> authorized_keys
chmod 600 authorized_keys
rm -fv id_rsa.pub

echo "ended setup ssh"

echo ""
echo ""
echo "init docker build"
cd
cd machine-learning
bash docker-build.sh mil DockerfileMiles
