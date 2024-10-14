from sklearn import svm

svm_model = svm.SVC(kernel='linear', class_weight='balanced')
svm_model.fit(X_train, Y_train)