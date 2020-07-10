#Definimos la clase ANN (Artificial Neural Network)

class ANN():
    
    def __init__(self, inputs, arch=None, activation=None, task=None):
        
        self.inputs=inputs               #Número de variables de entrada    
        self.task=task

        if arch!=None:
            self.architecture=arch            #Arquitectura global de la red (nº de capas y nº de neuronas por capa)
            self.activation=activation        #Función de activación para cada capa
            self.layers=len(arch)-1           #nº de capas de la red neuronal (sin contar la de entrada)
        
            #El tamaño de las listas debe coincidir
            if len(activation)!=self.layers:
                raise Exception("Number of activation functions not consistent with number of layers")
            
            #Inicialización de los pesos de la red
            import numpy as np
            
            layers=self.layers
            
            w=[0]*layers
            d=[0]*layers
            f=[0]*layers
            e=[0]*layers
            c=[0]*layers

            for l in range(layers):
                variables=arch[l]
                nodes=arch[l+1]
                w[l]=np.ones([variables+1,nodes])
                d[l]=np.zeros([variables+1,nodes])
                f[l]=np.zeros([variables+1,nodes])
                e[l]=np.ones([variables+1,nodes])*0.1
                c[l]=np.zeros([variables+1,nodes])
                
                #Asignamos valores randomizados para w
                for j in range(nodes):
                    if j%2==0:        #Valores Pares
                        w[l][0][j]=1
                    else:              #Valores impares
                        w[l][0][j]=-1
                    for i in range(variables):
                        w[l][i+1][j]=0.2*(np.random.rand()-0.5)   #Número aleatorio en [-0.1,0.1]              
                
            self.w=w #Pesos de la red
            self.d=d #Variables referentes al aprendizaje
            self.f=f
            self.e=e
            self.c=c
        else:
            self.architecture=[inputs]
            self.activation=[]
            self.layers=len(self.architecture)-1
            self.w=[]
            self.d=[]
            self.f=[]
            self.e=[]
            self.c=[]
        
    #Función para poder añadir más capas a la red
    def add_layer(self, neurons , function):
        
        import numpy as np
        self.architecture.append(neurons)       #Añadimos la capa con las neuronas correspondientes
        self.activation.append(function)        #Añadimos la función de activación de la capa
        self.layers=len(self.architecture)-1
        
        #Inicializamos los parametros correspondientes
        variables=self.architecture[-2]
        nodes=self.architecture[-1]
        
        self.w.append(np.ones([variables+1,nodes]))
        self.d.append(np.zeros([variables+1,nodes]))
        self.f.append(np.zeros([variables+1,nodes]))
        self.e.append(np.ones([variables+1,nodes])*0.1)
        self.c.append(np.zeros([variables+1,nodes]))
        
        #Valores para la inicialización (Igual que antes)
        for j in range(nodes):
            if j%2==0:
                self.w[-1][0][j]=1
            else:
                self.w[-1][0][j]=-1
            for i in range(variables):
                self.w[-1][i+1][j]=0.2*(np.random.rand()-0.5)
                        
    #Función para establecer los parámetros de entrenamiento
    def learning_options(self, cost_function, kappa, phi, theta, mu):
        
        self.cost_function=cost_function     #Añadimos una función de error con la que trabajar
        self.kappa=kappa
        self.phi=phi            #Parámetros del aprendizaje
        self.theta=theta
        self.mu=mu

    #Subrutina de preprocesado
    def preprocessing(self, data):

        from sklearn.preprocessing import LabelEncoder
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.preprocessing import MinMaxScaler 

        #Randomizamos las entradas de los datos
        data=data.sample(frac=1)

        if self.task=='Classification':

            x_data=data.iloc[:,:self.inputs].to_numpy() #Vector con los input en formato correcto
            t_data=data.iloc[:,self.inputs].to_numpy() #Vector con los target en formato array
            
            #Convertimos las variables categóricas en numéricas
            print('Converting categorical variables...')

            label_encoder=LabelEncoder()
            variables_salida=label_encoder.fit_transform(t_data)

            onehot_encoder=OneHotEncoder(sparse = False)
            variables_salida=variables_salida.reshape(len(variables_salida),1) 
            
            t_data=onehot_encoder.fit_transform(variables_salida)

            classes=[]
            for i in range(self.architecture[-1]):
                var=label_encoder.inverse_transform([i])
                print('Column:%i -->' %i, 'Class:%s' %var)
                classes.append(var)
            self.classes=classes

            #Normalizamos las variables numéricas
            print('Normalizing numerical values...')
            
            #Lo hacemos con la función del módulo porque no es necesario reescalar
            scaler = MinMaxScaler(feature_range=(0, 1))   
            x_data= scaler.fit_transform(x_data)  

            print('Done')    

            return x_data, t_data       

        elif self.task=='Regression':

            print('Normalizing numerical values...')

            data=data.to_numpy()

            n=np.size(data,0)
            m=np.size(data,1)

            new_data=np.zeros([n,m])

            maximums=np.zeros(m)
            minimums=np.zeros(m)

            for i in range(m): #Número de columnas
                maximum=max(data[:,i])
                minimum=min(data[:,i])
                aux=(data[:,i]-minimum)/(maximum-minimum)
                new_data[:,i]=aux
            
            self.scale_factor=(maximum,minimum) #Nos quedamos con el de la ultima columna

            x_data=new_data[:,:-1]

            t_data=new_data[:,-1]
            t_data=t_data.reshape(-1,1)

            print('Done')

            return x_data, t_data
            
        else:
            raise Exception('Task not recognized')

    #Entrenamiento de la red neuronal
    def train(self,x_train,t_train,epochs,batch_size=None):
        
        #Para llevar a cabo el entrenamiento necesitamos tres funciones: mapping, backpropagation, y changeweights
                
        #Comienza el entrenamiento
        if batch_size==None:
            batch_size=epochs

        entries=len(x_train)

        k=1 #Variable auxiliar para la barra de carga
        print('Training Neural Network ...')
        for m in range(epochs):
            #n=0
            error=0
            for n in range(entries):
                y_train=mapping(x_train,n,self.w,self.activation)
                self.d=backpropagation(y_train,t_train,n,self.w,self.d,self.activation,self.cost_function)
                
                if n%batch_size==0:
                    self.w,self.d,self.f,self.e,self.c=changeweight(self.w,self.d,self.f,self.e,self.c,
                                             self.kappa,self.phi,self.theta,self.mu)
                error=error+sum((y_train[-1]-t_train[n])**2)
                #n=n+1
            error=error/(entries*len(y_train[-1]))
            
            #m=m+1
            #Print de algunas magnitudes de interés (error, época, y porcentaje de ejecución)
            if m%(epochs/10)==0:
                string1=k*'#'
                string2=(10-k)*' '
                print('Error: %9.5f' %error,'%sEpoch: %5i' %('  ',m),'[%s%s]' %(string1,string2),'%i/%i' %(m,epochs))
                k=k+1
        #Fin del entrenamiento
        #====================================================================================================
    
    def predict(self, x_data, target=None):
        
        #Esta función simplemente hace el mapping para obtener una predicción       
        #Hacemos el mapeado con el vector que nos dan
        import numpy as np
        entries=len(x_data)
        variables=self.architecture[-1]
        y_predict=np.zeros([entries,variables])
        n=0
        for _ in range(entries):
            aux_1=mapping(x_data,n,self.w,self.activation)
            y_predict[n]=aux_1[-1]
            n=n+1

        if self.task=='Regression':

            try:
                maximum=self.scale_factor[0]
                minimum=self.scale_factor[1]
                #Reescalamos las variables a su valor original    
                y_result=np.zeros([entries,variables])
                y_result[:,0]=minimum+(maximum-minimum)*y_predict[:,0]
                return y_result
        
            except:
                return y_predict

        elif self.task=='Classification':
            
            if target==None or target=='probability':
                return y_predict

            elif target=='class':
                y_predict=np.round(y_predict) 
                y_result=[0]*entries
                for i in range(entries):
                    for j in range(variables):
                        if y_predict[i][j]==1:
                            y_result[i]=self.classes[j]
                return y_result

            else:
                raise Exception('Target not recognized')

          
        elif self.task==None:
            return y_predict

        else: 
            raise Exception('Task not recognized')

    def evaluate(self,x_data,t_data,epochs,batch_size=None,metric=None,folds=None):

        import numpy as np

        if batch_size==None:
            batch_size=epochs

        if metric==None:
            if self.task=='Regression':
                metric='MAE'
        
            elif self.task=='Classification':
                metric='accuracy'

            else:
                raise Exception('Please Specify Neural Network Task')

        if folds==None:
            folds=4

        entries=len(x_data)
        sample_entries=int(entries/folds)
        performances=[]*folds

        for k in range(folds):

            print('Processing Fold #', k+1)
            
            x_test=x_data[k*sample_entries:(k+1)*sample_entries]
            t_test=t_data[k*sample_entries:(k+1)*sample_entries]

            x_train=np.concatenate([x_data[:k*sample_entries],
                                    x_data[(k+1)*sample_entries:]],
                                    axis=0)

            t_train=np.concatenate([t_data[:k*sample_entries],
                                    t_data[(k+1)*sample_entries:]],
                                    axis=0)

            #Entrenamiento de la red neuronal
            #Comienza el entrenamiento

            entries=len(x_train)

            k=1 #Variable auxiliar para la barra de carga
            print('Training Neural Network ...')
            for m in range(epochs):
                #n=0
                error=0
                for n in range(entries):
                    y_train=mapping(x_train,n,self.w,self.activation)
                    self.d=backpropagation(y_train,t_train,n,self.w,self.d,self.activation,self.cost_function)
                    
                    if n%batch_size==0:
                        self.w,self.d,self.f,self.e,self.c=changeweight(self.w,self.d,self.f,self.e,self.c,
                                                self.kappa,self.phi,self.theta,self.mu)
                    error=error+sum((y_train[-1]-t_train[n])**2)
                    #n=n+1
                error=error/(entries*len(y_train[-1]))
                
                #m=m+1
                #Print de algunas magnitudes de interés (error, época, y porcentaje de ejecución)
                if m%(epochs/10)==0:
                    string1=k*'#'
                    string2=(10-k)*' '
                    print('Error: %9.5f' %error,'%sEpoch: %5i' %('  ',m),'[%s%s]' %(string1,string2),'%i/%i' %(m,epochs))
                    k=k+1
            #Fin del entrenamiento
            #Predicción con el test set
            print('Testing Neural Network...')

            entries=len(x_test)
            variables=self.architecture[-1]
            y_test=np.zeros([entries,variables])
            n=0
            for _ in range(entries):
                aux_1=mapping(x_test,n,self.w,self.activation)
                y_test[n]=aux_1[-1]
                n=n+1  

            #Evaluación de las Métricas (y_test,t_test)
            if self.task=='Regression':

                def mae(y,t):
                    n=len(t)
                    f=sum(abs(y-t))/n
                    return f[0]

                metric_list={'MAE':mae}
                
                performance=metric_list[metric](y_test,t_test)
                print(metric,'= %.4f' %performance)

                performances.append(performance)

            elif self.task=='Classification':

                def accuracy(y,t):
                    n=len(y)
                    y=np.round(y)
                    correct=0
                    for i in range(n):
                        if list(y[i])==list(t[i]):
                            correct+=1
                    return 1.0*correct/n
                
                metric_list={'accuracy':accuracy}

                performance=metric_list[metric](y_test,t_test)
                print(metric,'= %.4f' %performance)

                performances.append(performance)

            else:
                raise Exception('Please Specify Neural Network Task')

            #Resetear Pesos Red Neuronal
            layers=self.layers
            arch=self.architecture
                
            w=[0]*layers
            d=[0]*layers
            f=[0]*layers
            e=[0]*layers
            c=[0]*layers

            for l in range(layers):
                variables=arch[l]
                nodes=arch[l+1]
                w[l]=np.ones([variables+1,nodes])
                d[l]=np.zeros([variables+1,nodes])
                f[l]=np.zeros([variables+1,nodes])
                e[l]=np.ones([variables+1,nodes])*0.1
                c[l]=np.zeros([variables+1,nodes])
                
                #Asignamos valores randomizados para w
                for j in range(nodes):
                    if j%2==0:        #Valores Pares
                        w[l][0][j]=1
                    else:              #Valores impares
                        w[l][0][j]=-1
                    for i in range(variables):
                        w[l][i+1][j]=0.2*(np.random.rand()-0.5)   #Número aleatorio en [-0.1,0.1]              
                
            self.w=w #Pesos de la red
            self.d=d #Variables referentes al aprendizaje
            self.f=f
            self.e=e
            self.c=c

        #Hacemos la media de los rendimientos y hacemos print del resultado por pantalla
        performance=np.mean(performances)

        print('Final Result:',metric,'=',performance)    

#Subrutinas Utilizadas durante toda la ejecución del programa
#==================================================================================
def mapping(x,n,w,activation): 

    #Importación de paquetes necesarios
    import numpy as np

    def logistic(x):
        f=1/(1+np.exp(-x))
        return f

    def softmax(x): 
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def reLU(x):
        return np.maximum(0,x)


    activation_list = {'logistic': logistic,
                        'softmax' : softmax,
                        'reLU' : reLU}

    #Cálculo de las variables auxiliares
    layers=len(w) 
    y=[0]*(layers+1)
    y[0]=x[n]
    z=x[n]

    #Mapping

    for l in range(layers):
        nodes=np.size(w[l],1) 
        y[l+1]=np.zeros(nodes)
        variables=np.size(w[l],0)-1
        for j in range(nodes):
            y[l+1][j]=w[l][0][j]+sum(w[l][1:][:,j]*z) #Hace la multiplicacion de peso por variable y suma 
        y[l+1]=activation_list[activation[l]](y[l+1]) #w[l][1:][:,j] automaticamente en formato fila para poder multiplicar por z
        z=np.copy(y[l+1])

    #Fin de la subrutina
    #========================================================================================================
    return y

def backpropagation(y,t,n,w,deriv_total,activation,cost_function):
    #Comienzo de la subrutina
    #===================================================================================================

    #Importamos los módulos necesarios
    import numpy as np

    #Definimos algunas magnitudes que utilizaremos
    layers=len(w)
    delta=[0]*layers
    deriv=[0]*layers

    #Definimos las funciones que calculan la derivada de la función de activación
    def d_logistic(y):
        f=y*(1-y)
        f=np.diag(f)
        return f

    def d_reLU(y):
        f=np.heaviside(y,1)
        f=np.diag(f)
        return f

    def d_softmax(y):
        import numpy as np
        m=len(y)
        f=np.zeros([m,m])
        for i in range(m):
            for j in range(m):
                if j==i:
                    f[i,j]=y[i]*(1-y[i])
                else:
                    f[i,j]=-y[i]*y[j]
        return f

    #Diccionario con las funciones de activación posibles
    derivative_list = {'logistic': d_logistic,
                        'softmax' : d_softmax,
                        'reLU' : d_reLU}

    #Definimos las derivadas de las funciones de error
    def MSE(y,t,n):
        return y[-1]-t[n]

    def cross_entropy(y,t,n):
        f=1.0*(y[-1]-t[n])/(y[-1]*(1-y[-1]))
        return f

    #Diccionario con las derivadas del error dependiendo de la función utilizada para el mismo

    err_deriv_list = {'MSE' : MSE,
                        'cross_entropy' : cross_entropy}

    #Calculamos las derivadas para la última capa
    nodes=np.size(w[-1],1)
    variables=np.size(w[-1],0)-1
    deriv[-1]=np.zeros([variables+1,nodes])
    delta[-1]=np.zeros(nodes)

    #Delta de la capa de salida
    aux=derivative_list[activation[-1]](y[-1])
    aux2=err_deriv_list[cost_function](y,t,n)
    for j in range(nodes):
        delta[-1][j]=sum(aux2*aux[j])

    #Asignamos los valores a las derivadas 
    deriv[-1][0]=delta[-1]
    for i in range(variables):
        deriv[-1][i+1]=delta[-1]*y[-2][i]

    #Añadimos la derivada al total    
    deriv_total[-1]=deriv_total[-1]+deriv[-1]

    #Calculamos las derivadas para las demás capas iterativamente
    for l in reversed(range(layers-1)):
        nodes=np.size(w[l],1)
        variables=np.size(w[l],0)-1 

        deriv[l]=np.zeros([variables+1,nodes])
        delta[l]=np.zeros(nodes)

        #Cálculo del delta de la capa actual
        aux3=derivative_list[activation[l]](y[l+1])  #Elemento l+1 en y porque y[0]=x
        for j in range(nodes):
            delta[l][j]=sum(delta[l+1]*w[l+1][j+1])*sum(aux3[:,j])
            #Elemento j+1 porque empieza en variable no en bias

        #Asignamos los valores a las derivadas
        deriv[l][0]=delta[l]
        for i in range(variables):
            deriv[l][i+1]=delta[l]*y[l][i] #Elemento l en y porque y[0]=x

        #Añadimos la derivada al total
        deriv_total[l]=deriv_total[l]+deriv[l]
    #Fin de la subrutina
    #==========================================================================================================
    return deriv_total

def changeweight(w,d,f,e,c,kappa,phi,theta,mu):
    #Subrutina cambio de pesos

    #Importamos los paquetes necesarios
    import numpy as np

    #Parámetros de utilidad
    layers=len(w)

    #Iteramos para cambiar todos los pesos
    for l in range(layers):
        nodes=np.size(w[l],1)
        variables=np.size(w[l],0)-1
        for j in range(nodes):
            for i in range(variables+1):

                if d[l][i][j]*f[l][i][j]>0:
                    e[l][i][j]=e[l][i][j]+kappa
                else:
                    e[l][i][j]=e[l][i][j]*phi

                f[l][i][j]=(1-theta)*d[l][i][j]+theta*f[l][i][j]
                c[l][i][j]=(1-mu)*(-1)*e[l][i][j]*d[l][i][j]+mu*c[l][i][j]
                w[l][i][j]=w[l][i][j]+c[l][i][j]
                d[l][i][j]=0

    return w,d,f,e,c  
#===================================================================================
