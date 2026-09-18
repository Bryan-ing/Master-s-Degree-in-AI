import os
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Datos de ejemplo: coordenadas (x, y) de defectos detectados sobre el mapa de una oblea
#estos puntos representarían las posiciones de los dies marcados como "fail"
#dentro del wafer map (en vez de puntos genéricos, aquí son ubicaciones reales de cada defecto)
X = np.array([
    [2, 2],
    [2.3, 1.8],
    [1.8, 2.4],
    [20, 20],
    [21, 19],
    [19.5, 20.5],
    [2, 20],
    [1.5, 19],
    [2.5, 20.8],
    [20, 2],
    [21, 1.5],
    [19, 2.5]
])

# Crear el modelo K-Means con 3 clusters
kmeans = KMeans(n_clusters=4, random_state=42)  #el n_clusters dice encuentra 4 grupos en esos datos
#cada grupo representaría una zona del wafer donde se concentran los defectos
#(ej. una esquina, un borde, el centro), útil para detectar patrones de falla
#el random_state es solamente la semilla para que empiece con el mismo número

# Entrenar el modelo
kmeans.fit(X)  #coloca los 4 centroides-calcula para cada punto
#a que centroide esta mas cerca y lo asigna a un grupo
#recalcula cada centroide como el promedio de todos los puntos que le tocaron
#repite los pasos hasta que el centroide no se mueve

# Etiquetas asignadas a cada punto
labels = kmeans.labels_ #dice a que grupo (zona del wafer) quedo asignado cada defecto

# Centroides
centroids = kmeans.cluster_centers_ #te da las coordenadas finales de
#los 4 centroides despues de converger, es decir, el centro de cada zona de defectos

print("Etiquetas:", labels)
print("Centroides:")
print(centroids)

# Visualización
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
#grafica los defectos coloreados según la zona/grupo c=labels y
#marca con una X roja grande el centro de cada zona de concentración de defectos
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker='X',
    s=200,
    c='red',
    label='Centroides'
)

plt.title('K-Means Clustering - Zonas de defecto en wafer (WaferPi)')
plt.xlabel('Posición X en el wafer')
plt.ylabel('Posición Y en el wafer')
plt.legend()
plt.show()
