export JAVA_HOME=~/Downloads/jdk-12.0.2
cat src/main/resources/levels/generated/basic/Easy_1/0_Basic_Easy_1.lvl | ~/apache-maven-3.5.0/bin/mvn clean compile exec:java -Dexec.mainClass="searchclient.SearchClient" -Dexec.args="-alpha -train -python python3 -gpus 2 -generate 10 basic easy_1"
