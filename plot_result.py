import matplotlib.pyplot as plt
import pandas as pd

results = pd.read_csv('./train_hist.csv')
# list all data in history

# summarize history for accuracy
plt.plot(results['accuracy'], 'b')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.xlim(0, len(results))
plt.show()

# summarize history for loss
plt.plot(results['loss'], 'r')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xlim(0, len(results))
plt.show()