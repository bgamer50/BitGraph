pipeline {
  agent any
  stages {
    stage('Build') {
      steps {
        sh 'make components.exe'
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