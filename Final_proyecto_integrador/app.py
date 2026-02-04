"""
Aplicación Web para Sistema de Clustering Online
================================================
Interfaz Flask para clasificar imágenes usando clustering online
"""

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import os
from pathlib import Path
import base64
from clustering_online_completo import ClusteringOnlineSystem, OnlineDynamicClustering
import random
import json
import threading
import queue
import numpy as np
import gc  # Garbage collector para liberar memoria

app = Flask(__name__)

# Configuración - TODO EN MEMORIA
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'avif', 'webp'}

# Aumentar límites para soportar grandes cantidades de imágenes (hasta 5GB)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB max


# Manejador de error 413 (Request Entity Too Large)
@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({
        'error': 'El tamaño total de los archivos es demasiado grande. Intenta subir menos imágenes a la vez (máximo ~500 imágenes por lote).',
        'sugerencia': 'Divide las imágenes en grupos más pequeños y súbelas por partes.'
    }), 413

# Cache de sistemas de clustering
sistema_online = None

# Estado para modo Online LIVE
online_live_state = {
    'running': False,
    'thread': None,
    'input_queue': None,
    'output_queue': None,
    'stop_event': None,
    'lock': threading.Lock(),
    'system': None,
    'model': None,
    'metodo': None,
    'n_clusters': None,
    'capacities': None,
    'total_images': None,
    'processed': 0
}

def get_sistema_online(metodo, n_clusters, cluster_sizes=None, distance_threshold=0.65):
    """Obtiene o crea el sistema de clustering online único (TODO EN MEMORIA)"""
    global sistema_online
    # Verificar si necesitamos crear nuevo sistema (incluyendo si cambió el umbral)
    if (sistema_online is None or 
        sistema_online.metodo != metodo or 
        sistema_online.n_clusters != n_clusters or
        sistema_online.cluster_sizes != cluster_sizes or
        getattr(sistema_online, 'distance_threshold', None) != distance_threshold): # <--- CAMBIO: Chequeo de umbral
        
        # NO usar storage_dir - todo se mantiene en memoria
        sistema_online = ClusteringOnlineSystem(
            metodo=metodo, 
            n_clusters=n_clusters, 
            cluster_sizes=cluster_sizes,
            storage_dir=None,  # No se usa, todo en memoria
            distance_threshold=distance_threshold # <--- CAMBIO: Pasar umbral
        )
    return sistema_online

def allowed_file(filename):
    """Verifica si el archivo tiene extensión permitida"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(image_bytes):
    """Convierte bytes de imagen a base64 para mostrar en web"""
    try:
        if isinstance(image_bytes, bytes):
            return base64.b64encode(image_bytes).decode('utf-8')
        return None
    except:
        return None

def _init_live_model(system, n_clusters):
    """Inicializa el modelo online con capacidades amplias."""
    capacities = online_live_state.get('capacities') or ([10**12] * n_clusters)
    # Recuperar umbral del estado global
    threshold = online_live_state.get('threshold', 0.65) # <--- CAMBIO: Leer del estado
    return OnlineDynamicClustering(max_k=n_clusters, capacities=capacities, threshold=threshold)

def _live_worker():
    """Worker persistente para modo Online LIVE."""
    state = online_live_state
    input_q = state['input_queue']
    output_q = state['output_queue']
    stop_event = state['stop_event']
    system = state['system']

    while not stop_event.is_set():
        try:
            item = input_q.get(timeout=0.2)
        except queue.Empty:
            continue

        if item is None:
            input_q.task_done()
            break

        try:
            img_bytes = item['bytes']
            filename = item['filename']
            label = item.get('label', None)

            with state['lock']:
                if state['total_images'] is not None and state['processed'] >= state['total_images']:
                    output_q.put({'type': 'error', 'error': 'Capacidad total alcanzada', 'filename': filename})
                    continue

                if state['model'] is None:
                    state['model'] = _init_live_model(system, state['n_clusters'])

                if sum(state['model'].remaining) <= 0:
                    output_q.put({'type': 'error', 'error': 'Todos los clusters están llenos', 'filename': filename})
                    continue

                img_prep = system.preprocesar_imagen(img_bytes)
                features_raw = system.extraer_caracteristicas(img_prep)
                features_norm = system.normalizar_l2(features_raw)

                cluster = state['model'].partial_fit(features_norm)
                dist = float(np.linalg.norm(features_norm - state['model'].cluster_centers_[cluster]))

                system.X_accumulated.append(features_raw)
                system.labels_accumulated.append(label)
                system.filenames_accumulated.append(filename)
                system.predictions_accumulated.append(int(cluster))
                system.has_data = True
                state['processed'] += 1

                # Métricas internas en tiempo real (externas se reservan para stop)
                X_all_norm = np.array([system.normalizar_l2(x) for x in system.X_accumulated])
                predicciones = np.array(system.predictions_accumulated)
                metricas_all = system._calcular_metricas(X_all_norm, predicciones)
                metricas_all['externas'] = {}

                result_data = {
                    'type': 'result',
                    'data': {
                        'nombre': filename,
                        'cluster': int(cluster),
                        'distancia': round(dist, 4),
                        'confianza': round((1.0 / (1.0 + dist)) * 100, 2),
                        'label_real': label,
                        'imagen': encode_image_to_base64(img_bytes)
                    },
                    'metricas': metricas_all,
                    'total_acumulado': len(system.X_accumulated),
                    'distribucion': state['model'].get_distribution()
                }

            output_q.put(result_data)

        except Exception as e:
            output_q.put({'type': 'error', 'error': str(e), 'filename': item.get('filename', 'unknown')})
        finally:
            input_q.task_done()

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/api/info')
def get_info():
    """Devuelve información sobre métodos disponibles"""
    info = {
        'metodos': [
            {
                'id': 'hog',
                'name': 'HOG',
                'descripcion': 'Histograma de Gradientes',
                'velocidad': 'Rápida'
            },
            {
                'id': 'sift',
                'name': 'SIFT',
                'descripcion': 'Descriptores locales',
                'velocidad': 'Media'
            },
            {
                'id': 'hu_moments',
                'name': 'Hu Moments',
                'descripcion': 'Momentos invariantes',
                'velocidad': 'Muy rápida'
            },
            {
                'id': 'cnn',
                'name': 'CNN (VGG16)',
                'descripcion': 'Red neuronal profunda',
                'velocidad': 'Lenta'
            }
        ]
    }
    return jsonify(info)

@app.route('/api/classify_stream', methods=['POST'])
def classify_stream():
    """Clasifica imágenes con streaming en tiempo real (Server-Sent Events)"""
    
    def generate():
        try:
            yield f"data: {json.dumps({'type': 'connected', 'message': 'Conexión establecida'})}\n\n"
            
            if 'files' not in request.files:
                yield f"data: {json.dumps({'error': 'No se enviaron archivos'})}\n\n"
                return
            
            metodo = request.form.get('metodo', 'hog')
            n_clusters = int(request.form.get('n_clusters', 3))
            realtime = str(request.form.get('realtime', 'false')).lower() == 'true'
            cluster_sizes_str = request.form.get('cluster_sizes', None)
            
            # --- CAMBIO: Obtener umbral del request (default 0.65) ---
            try:
                distance_threshold = float(request.form.get('threshold', 0.65))
            except (ValueError, TypeError):
                distance_threshold = 0.65
            # ---------------------------------------------------------

            # Parsear tamaños exactos de clusters
            cluster_sizes = None
            if cluster_sizes_str:
                try:
                    cluster_sizes = [int(x.strip()) for x in cluster_sizes_str.split(',')]
                    if len(cluster_sizes) != n_clusters:
                        yield f"data: {json.dumps({'type': 'error', 'error': f'Especificaste {len(cluster_sizes)} tamaños pero hay {n_clusters} clusters'})}\n\n"
                        return
                except:
                    yield f"data: {json.dumps({'type': 'error', 'error': 'Formato inválido para cluster_sizes. Usa: 5,7,8'})}\n\n"
                    return
            
            files = request.files.getlist('files')
            labels_json = request.form.get('labels', '[]')
            
            if not files or files[0].filename == '':
                yield f"data: {json.dumps({'error': 'No se seleccionaron archivos'})}\n\n"
                return
            
            labels_dict = {}
            try:
                labels_dict = json.loads(labels_json)
            except:
                pass
            
            imagenes_data = []
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    image_bytes = file.read()
                    label = labels_dict.get(filename, None)
                    imagenes_data.append({'bytes': image_bytes, 'label': label, 'filename': filename})
            
            if not imagenes_data:
                yield f"data: {json.dumps({'error': 'No se procesaron archivos válidos'})}\n\n"
                return

            random.shuffle(imagenes_data)
            
            if cluster_sizes:
                if sum(cluster_sizes) != len(imagenes_data):
                    if realtime:
                        cluster_sizes = None
                    else:
                        yield f"data: {json.dumps({'type': 'error', 'error': f'La suma de tamaños ({sum(cluster_sizes)}) no coincide con el total de imágenes ({len(imagenes_data)})'})}\n\n"
                        return
            
            # --- CAMBIO: Pasar threshold al obtener sistema ---
            sistema = get_sistema_online(metodo, n_clusters, cluster_sizes, distance_threshold=distance_threshold)
            
            for evento in sistema.procesar_imagenes_streaming(imagenes_data, realtime=realtime):
                if evento['type'] == 'start':
                    yield f"data: {json.dumps({'type': 'start', 'total': evento['total']})}\n\n"
                elif evento['type'] == 'progress':
                    res = evento['resultado']
                    progress_data = {
                        'type': 'progress',
                        'current': evento['current'],
                        'total': evento['total'],
                        'filename': res['nombre'],
                        'cluster_distribution': evento['cluster_distribution']
                    }
                    yield f"data: {json.dumps(progress_data)}\n\n"
                    img_base64 = encode_image_to_base64(res['image_bytes'])
                    result_data = {
                        'type': 'result',
                        'data': {
                            'nombre': res['nombre'],
                            'cluster': res['cluster'],
                            'distancia': round(res['distancia'], 4),
                            'confianza': round(res['confianza'] * 100, 2),
                            'label_real': res['label_real'],
                            'imagen': img_base64
                        }
                    }
                    yield f"data: {json.dumps(result_data)}\n\n"
                elif evento['type'] == 'complete':
                    complete_data = {
                        'type': 'complete',
                        'total': evento['total'],
                        'metricas': evento['metricas'],
                        'total_acumulado': evento['total_acumulado'],
                        'distribucion_final': evento['distribucion_final']
                    }
                    yield f"data: {json.dumps(complete_data)}\n\n"
                elif evento['type'] == 'error' or evento['type'] == 'image_error':
                    yield f"data: {json.dumps(evento)}\n\n"
            
            del imagenes_data
            gc.collect()
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            gc.collect()
            error_msg = f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            yield error_msg
    
    response = Response(stream_with_context(generate()), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response

# ======================== ONLINE LIVE ========================

@app.route('/api/online/start', methods=['POST'])
@app.route('/api/online/start', methods=['POST'])
def online_start():
    """Inicia el modo Online LIVE con threshold configurable"""
    state = online_live_state
    with state['lock']:
        if state['running']:
            return jsonify({'error': 'Online LIVE ya está en ejecución'}), 400

        data = request.json if request.is_json else request.form
        metodo = data.get('metodo', 'hog')
        n_clusters = int(data.get('n_clusters', 3))
        total_images = data.get('total_images', None)
        cluster_sizes_raw = data.get('cluster_sizes', None)
        
        # --- CAMBIO: Obtener umbral ---
        try:
            threshold = float(data.get('threshold', 0.65))
        except (ValueError, TypeError):
            threshold = 0.65
        # -----------------------------

        cluster_sizes = None
        if cluster_sizes_raw:
            try:
                cluster_sizes = [int(x.strip()) for x in str(cluster_sizes_raw).split(',')]
            except Exception:
                return jsonify({'error': 'Formato inválido para cluster_sizes. Usa: 5,7,8'}), 400

        if cluster_sizes:
            if len(cluster_sizes) != n_clusters:
                return jsonify({'error': 'cluster_sizes no coincide con n_clusters'}), 400
            if any(x <= 0 for x in cluster_sizes):
                return jsonify({'error': 'cluster_sizes debe tener valores > 0'}), 400
            total_images = sum(cluster_sizes)
            capacities = cluster_sizes
        else:
            total_images = int(total_images) if total_images is not None else None
            if total_images is None or total_images <= 0:
                return jsonify({'error': 'total_images es obligatorio para LIVE'}), 400
            if total_images < n_clusters:
                return jsonify({'error': 'total_images debe ser >= n_clusters'}), 400

            base = total_images // n_clusters
            resto = total_images % n_clusters
            capacities = [base + (1 if i < resto else 0) for i in range(n_clusters)]

        # --- CAMBIO: Inicializar sistema con el umbral ---
        state['system'] = ClusteringOnlineSystem(
            metodo=metodo, 
            n_clusters=n_clusters, 
            cluster_sizes=None, 
            storage_dir=None,
            distance_threshold=threshold # <--- AQUI
        )
        
        state['model'] = None
        state['input_queue'] = queue.Queue()
        state['output_queue'] = queue.Queue()
        state['stop_event'] = threading.Event()
        state['metodo'] = metodo
        state['n_clusters'] = n_clusters
        state['capacities'] = capacities
        state['total_images'] = total_images
        state['processed'] = 0
        state['threshold'] = threshold # <--- CAMBIO: Guardar umbral en estado LIVE

        state['running'] = True
        state['thread'] = threading.Thread(target=_live_worker, daemon=True)
        state['thread'].start()

    return jsonify({'success': True, 'message': 'Online LIVE iniciado', 'metodo': metodo, 'n_clusters': n_clusters, 'capacities': capacities, 'threshold': threshold})
@app.route('/api/online/push', methods=['POST'])
def online_push():
    """Envía una imagen a la cola del modo Online LIVE"""
    state = online_live_state
    if not state['running']:
        return jsonify({'error': 'Online LIVE no está iniciado'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No se envió archivo'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'Archivo inválido'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Extensión no permitida'}), 400

    filename = secure_filename(file.filename)
    image_bytes = file.read()
    label = request.form.get('label', None)

    state['input_queue'].put({
        'bytes': image_bytes,
        'filename': filename,
        'label': label
    })

    return jsonify({'success': True, 'queued': True, 'filename': filename})

@app.route('/api/online/stream', methods=['GET'])
def online_stream():
    """SSE para resultados del modo Online LIVE"""
    state = online_live_state
    if not state['running']:
        return Response("data: {\"type\": \"error\", \"error\": \"Online LIVE no está iniciado\"}\n\n", mimetype='text/event-stream')

    def generate():
        yield f"data: {json.dumps({'type': 'connected'})}\n\n"
        while state['running'] or not state['output_queue'].empty():
            try:
                msg = state['output_queue'].get(timeout=0.5)
                yield f"data: {json.dumps(msg)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    response = Response(stream_with_context(generate()), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Connection'] = 'keep-alive'
    return response

@app.route('/api/online/stop', methods=['POST'])
def online_stop():
    """Detiene el modo Online LIVE y retorna métricas finales"""
    state = online_live_state
    with state['lock']:
        if not state['running']:
            return jsonify({'error': 'Online LIVE no está iniciado'}), 400

        state['running'] = False
        if state['stop_event']:
            state['stop_event'].set()
        if state['input_queue']:
            state['input_queue'].put(None)

    if state['thread']:
        state['thread'].join(timeout=2)

    metricas_finales = {'internas': {}, 'externas': {}}
    if state['system'] and state['system'].has_data:
        X_all_norm = np.array([state['system'].normalizar_l2(x) for x in state['system'].X_accumulated])
        predicciones = np.array(state['system'].predictions_accumulated)
        metricas_finales = state['system']._calcular_metricas(X_all_norm, predicciones)

    with state['lock']:
        state['system'] = None
        state['model'] = None
        state['input_queue'] = None
        state['output_queue'] = None
        state['stop_event'] = None
        state['thread'] = None
        state['metodo'] = None
        state['n_clusters'] = None
        state['capacities'] = None
        state['total_images'] = None
        state['processed'] = 0

    gc.collect()
    return jsonify({'success': True, 'message': 'Online LIVE detenido', 'metricas': metricas_finales})

@app.route('/api/reset', methods=['POST'])
def reset_sistema():
    """Reinicia el sistema de clustering online y libera memoria"""
    global sistema_online
    try:
        metodo = request.json.get('metodo', 'hog')
        n_clusters = int(request.json.get('n_clusters', 3))
        cluster_sizes_str = request.json.get('cluster_sizes', None)
        
        # Parsear tamaños si se especificaron
        cluster_sizes = None
        if cluster_sizes_str:
            try:
                cluster_sizes = [int(x.strip()) for x in cluster_sizes_str.split(',')]
            except:
                pass
        
        # Liberar sistema anterior completamente
        if sistema_online is not None:
            sistema_online.reset_sistema()
            sistema_online = None
        
        # Forzar liberación de memoria
        gc.collect()
        
        return jsonify({'success': True, 'message': 'Sistema reiniciado correctamente'})
    except Exception as e:
        gc.collect()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("SISTEMA DE CLUSTERING ONLINE INCREMENTAL")
    print("="*70)
    print("\n🚀 Iniciando servidor Flask...")
    print("✓ Procesamiento 100% en memoria (sin archivos en disco)")
    print("✓ Límite de tamaño: 5GB (soporta cientos de imágenes)\n")
    
    print("\n📋 Instrucciones:")
    print("   1. Abre tu navegador en: http://localhost:5000")
    print("   2. Selecciona método y número de clusters")
    print("   3. Sube imágenes (opcionalmente con etiquetas)")
    print("   4. El sistema aprenderá progresivamente")
    print("   5. Observa métricas de evaluación en tiempo real\n")
    print("="*70)
    print("Presiona Ctrl+C para detener el servidor\n")
    
    # Puerto 7860 para Hugging Face Spaces, 5000 para local
    port = int(os.environ.get('PORT', 7860))
    
    # debug=False para mejor rendimiento con muchas imágenes
    # threaded=True para manejar múltiples conexiones
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)