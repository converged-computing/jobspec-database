export JAVA_HOME=~/Downloads/jdk-12.0.2
java=~/Downloads/jdk-12.0.2/bin/java
~/apache-maven-3.5.0/bin/mvn clean install
mkdir -p logs/nn/basic
mkdir -p logs/nn/randomwalk
mkdir -p logs/nn/dungeon

rm -rf logs/nn/basic/*
rm -rf logs/nn/randomwalk/*
rm -rf logs/nn/dungeon/*

$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx70G -classpath target/classes searchclient.SearchClient -alpha -best -python python3 -nolimit" -l src/main/resources/levels/generated/basic/Easy_1 -t 1200 -o logs/nn/basic/Easy.zip
$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx70G -classpath target/classes searchclient.SearchClient -alpha -best -python python3 -nolimit" -l src/main/resources/levels/generated/basic/Easy_2 -t 1200 -o logs/nn/basic/Medium.zip
$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx70G -classpath target/classes searchclient.SearchClient -alpha -best -python python3 -nolimit" -l src/main/resources/levels/generated/basic/Easy_3 -t 1200 -o logs/nn/basic/Hard.zip

$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx70G -classpath target/classes searchclient.SearchClient -alpha -best -python python3 -nolimit" -l src/main/resources/levels/generated/randomwalk/Easy_1 -t 1200 -o logs/nn/randomwalk/Easy.zip
$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx70G -classpath target/classes searchclient.SearchClient -alpha -best -python python3 -nolimit" -l src/main/resources/levels/generated/randomwalk/Easy_2 -t 1200 -o logs/nn/randomwalk/Medium.zip
$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx70G -classpath target/classes searchclient.SearchClient -alpha -best -python python3 -nolimit" -l src/main/resources/levels/generated/randomwalk/Easy_3 -t 1200 -o logs/nn/randomwalk/Hard.zip

$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx70G -classpath target/classes searchclient.SearchClient -alpha -best -python python3 -nolimit" -l src/main/resources/levels/generated/dungeon/Easy_1 -t 1200 -o logs/nn/dungeon/Easy.zip
$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx70G -classpath target/classes searchclient.SearchClient -alpha -best -python python3 -nolimit" -l src/main/resources/levels/generated/dungeon/Easy_2 -t 1200 -o logs/nn/dungeon/Medium.zip
$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx70G -classpath target/classes searchclient.SearchClient -alpha -best -python python3 -nolimit" -l src/main/resources/levels/generated/dungeon/Easy_3 -t 1200 -o logs/nn/dungeon/Hard.zip
