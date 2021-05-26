docker rmi -f conda
docker build -t conda .
docker run conda

pause