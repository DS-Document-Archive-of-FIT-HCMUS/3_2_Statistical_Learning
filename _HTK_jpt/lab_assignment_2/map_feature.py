import numpy as np 
import json

# Hàm sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hàm tính cost
def compute_cost(X, y, theta, lambda_):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    reg_term = (lambda_ / (2 * m)) * np.sum(theta[1:]**2)  # regularization term, excluding theta[0]
    return cost + reg_term

# Hàm tính gradient
def compute_gradient(X, y, theta, lambda_):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    grad = (1 / m) * np.dot(X.T, (h - y))
    reg_term = (lambda_ / m) * theta
    reg_term[0] = 0  # No regularization for bias term
    return grad + reg_term

# Gradient Descent
def gradient_descent(X, y, theta, alpha, lambda_, num_iters):
    m = len(y)
    J_history = []

    for _ in range(num_iters):
        gradient = compute_gradient(X, y, theta, lambda_)
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta, lambda_)
        J_history.append(cost)

    return theta, J_history

# Dự đoán nhãn
def predict(X, theta):
    h = sigmoid(np.dot(X, theta))
    return (h >= 0.5).astype(int)

# Đánh giá mô hình
def evaluate(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score}

# Đọc cấu hình từ file JSON
def read_config(filename):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config

# Đọc dữ liệu từ file training_data.txt
def read_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

# Hàm lưu model vào file JSON
def save_model(filename, model):
    with open(filename, 'w') as f:
        json.dump(model, f)

# Hàm lưu kết quả đánh giá vào file JSON
def save_classification_report(filename, report):
    with open(filename, 'w') as f:
        json.dump(report, f)

# Hàm chính
def main():
    # Đọc cấu hình huấn luyện
    config = read_config('config.json')
    alpha = config['Alpha']
    lambda_ = config['Lambda']
    num_iters = config['NumIter']
    
    # Đọc dữ liệu huấn luyện
    X, y = read_data('training_data.txt')
    
    # Ánh xạ dữ liệu sang miền dữ liệu mới gồm 28 chiều
    X_mapped = map_feature(X[:,0], X[:,1])
    
    # Khởi tạo theta
    theta = np.zeros(X_mapped.shape[1])
    
    # Huấn luyện mô hình
    theta, J_history = gradient_descent(X_mapped, y, theta, alpha, lambda_, num_iters)
    
    # Lưu mô hình
    save_model('model.json', {'theta': theta.tolist()})
    
    # Dự đoán và đánh giá kết quả
    y_pred = predict(X_mapped, theta)
    report = evaluate(y, y_pred)
    
    # Lưu kết quả đánh giá
    save_classification_report('classification_report.json', report)

def map_feature(x1, x2):
#   x1, x2 type: numpy array
#   Returns a new feature array with more features, comprising of 
#   x1, x2, x1.^2, x2.^2, x1*x2, x1*x2.^2, etc.

    degree = 6
    out = np.ones([len(x1), int((degree + 1) * (degree + 2) / 2)])
    idx = 1

    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            a1 = x1 ** (i - j)
            a2 = x2 ** j
            out[:, idx] = a1 * a2
            idx += 1

    return out

if __name__ == '__main__':
    main()