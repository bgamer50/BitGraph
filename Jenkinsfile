pipeline {
  agent any
  stages {
    stage('Build') {
      steps {
        sh '''rm -rf ../gremlin++
git clone https://github.com/bgamer50/gremlin- -b master-dev ../gremlin++

cppcheck --enable=all --xml --xml-version=2 -I../gremlin++ .
make components.exe'''
        recordIssues(enabledForFailure: true, aggregatingResults: true, tools: [gcc(), cppCheck()])
      }
    }

    stage('Test') {
      steps {
        sh '''wget https://snap.stanford.edu/data/facebook_combined.txt.gz
gunzip facebook_combined.txt.gz
./components.exe facebook_combined.txt'''
      }
    }

  }
}