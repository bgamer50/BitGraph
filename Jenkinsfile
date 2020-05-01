pipeline {
  agent any
  stages {
    stage('Build') {
      steps {
        sh '''rm -rf ../gremlin++
git clone https://github.com/bgamer50/gremlin- -b master-dev ../gremlin++

cppcheck --enable-all .
make components.exe'''
        recordIssues()
        publishCppcheck()
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