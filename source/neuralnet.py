import tensorflow as tf

class CVAE(object):

    def __init__(self, height, width, channel, z_dim, mx, mz, leaning_rate=1e-3):

        print("\nInitializing Neural Network...")
        self.height, self.width, self.channel = height, width, channel
        self.k_size, self.z_dim = 3, z_dim
        self.mx, self.mz, = mx, mz
        self.leaning_rate = leaning_rate

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.batch_size = tf.placeholder(tf.int32, shape=[])

        self.weights, self.biasis = [], []
        self.w_names, self.b_names = [], []
        self.fc_shapes, self.conv_shapes = [], []
        self.features_r, self.features_f = [], []

        self.z_pack, self.z_r_pack, self.z_T_pack, self.z_Tr_pack, self.x_r, self.x_Tr =\
            self.build_model(input=self.x, ksize=self.k_size)

        """Loss of Transformer"""
        # self.T_term1 = tf.math.log(((self.z_T_pack[2] + 1e-12) / (self.z_pack[2] + 1e-12)) + 1e-12)
        # self.T_term2 = (tf.square(self.z_pack[2]) + tf.square(self.z_pack[1] - self.z_T_pack[1]) + 1e-12) / (2 * tf.square(self.z_T_pack[2]) + 1e-12)
        # self.T_term3 = - 0.5
        # self.loss_T = self.T_term1 + self.T_term2 + self.T_term3
        self.T_term1 = self.z_T_pack[2] * self.z_pack[2]
        self.T_term2 = tf.square(self.z_T_pack[1] - self.z_T_pack[2])
        self.T_term3 = tf.math.log(((self.z_T_pack[2] + 1e-12) / (self.z_pack[2] + 1e-12)) + 1e-12)
        self.loss_T = 0.5 * (self.T_term1 + self.T_term2 + self.T_term3 - self.z_dim)

        """Loss of Generator"""
        self.mse_r = self.mean_square_error(x1=self.x, x2=self.x_r)
        self.mse_Tr = self.mean_square_error(x1=self.x_r, x2=self.x_Tr)
        self.kld_r = self.kl_divergence(mu=self.z_r_pack[1], sigma=self.z_r_pack[2])
        self.kld_Tr = self.kl_divergence(mu=self.z_Tr_pack[1], sigma=self.z_Tr_pack[2])
        self.loss_G_z = self.mse_r + self.kld_r
        self.loss_G_zT = self.max_with_positive_margin(margin=self.mx, loss=self.mse_Tr) + \
            self.max_with_positive_margin(margin=self.mz, loss=self.kld_Tr)
        self.loss_G = self.loss_G_z + self.loss_G_zT

        """Loss of Encoder"""
        self.kld = self.kl_divergence(mu=self.z_pack[1], sigma=self.z_pack[2])
        self.loss_E = self.kld + \
            self.max_with_positive_margin(margin=self.mz, loss=self.kld_r) + \
            self.max_with_positive_margin(margin=self.mz, loss=self.kld_Tr)

        self.loss_T_mean = tf.compat.v1.reduce_mean(self.loss_T)
        self.loss_G_mean = tf.compat.v1.reduce_mean(self.loss_G)
        self.loss_E_mean = tf.compat.v1.reduce_mean(self.loss_E)

        self.loss1 = tf.compat.v1.reduce_mean(self.loss_T + self.loss_G)
        self.loss2 = self.loss_E_mean
        self.loss_tot = self.loss1 + self.loss2

        self.optimizer1 = tf.compat.v1.train.AdamOptimizer( \
            self.leaning_rate, beta1=0.5, beta2=0.999).minimize(self.loss1)
        self.optimizer2 = tf.compat.v1.train.AdamOptimizer( \
            self.leaning_rate, beta1=0.5, beta2=0.999).minimize(self.loss2)

        tf.compat.v1.summary.scalar('loss_T', self.loss_T_mean)
        tf.compat.v1.summary.scalar('loss_G', self.loss_G_mean)
        tf.compat.v1.summary.scalar('loss_E(loss_2)', self.loss_E_mean)
        tf.compat.v1.summary.scalar('loss_1(T&G)', self.loss1)
        tf.compat.v1.summary.scalar('loss_tot', self.loss_tot)
        self.summaries = tf.compat.v1.summary.merge_all()

    def mean_square_error(self, x1, x2):

        data_dim = len(x1.shape)
        if(data_dim == 4):
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2), axis=(1, 2, 3))
        elif(data_dim == 3):
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2), axis=(1, 2))
        elif(data_dim == 2):
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2), axis=(1))
        else:
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2))

    def kl_divergence(self, mu, sigma):

        return 0.5 * tf.compat.v1.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.math.log(tf.square(sigma) + 1e-12) - 1, axis=(1))

    def max_with_positive_margin(self, margin, loss):


        return tf.compat.v1.math.maximum(x=tf.compat.v1.zeros_like(loss), y=(tf.compat.v1.ones_like(loss) * margin) - loss)

    def build_model(self, input, ksize=3):

        # with tf.name_scope('encoder') as scope_enc:
        z, z_mu, z_sigma = self.encoder(input=input, ksize=ksize)
        x_r = self.generator(input=z, ksize=ksize)
        z_r, z_mu_r, z_sigma_r = self.encoder(input=x_r, ksize=ksize)

        z_T = self.transformer(input=z)
        x_Tr = self.generator(input=z_T, ksize=ksize)
        z_Tr, z_mu_Tr, z_sigma_Tr = self.encoder(input=x_Tr, ksize=ksize)

        z_pack = [z, z_mu, z_sigma]
        z_r_pack = [z_r, z_mu_r, z_sigma_r]
        z_T_pack = [z_r, z_mu_r, z_sigma_r]
        z_Tr_pack = [z_Tr, z_mu_Tr, z_sigma_Tr]

        return z_pack, z_r_pack, z_T_pack, z_Tr_pack, x_r, x_Tr

    def encoder(self, input, ksize=3):

        print("\nEncode-1")
        conv1_1 = self.conv2d(input=input, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 1, 16], activation="elu", name="endconv1_1")
        conv1_2 = self.conv2d(input=conv1_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 16], activation="elu", name="endconv1_2")
        maxp1 = self.maxpool(input=conv1_2, ksize=2, strides=2, padding='SAME', name="endmax_pool1")
        self.conv_shapes.append(conv1_2.shape)

        print("Encode-2")
        conv2_1 = self.conv2d(input=maxp1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 32], activation="elu", name="endconv2_1")
        conv2_2 = self.conv2d(input=conv2_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 32], activation="elu", name="endconv2_2")
        maxp2 = self.maxpool(input=conv2_2, ksize=2, strides=2, padding='SAME', name="endmax_pool2")
        self.conv_shapes.append(conv2_2.shape)

        print("Encode-3")
        conv3_1 = self.conv2d(input=maxp2, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 64], activation="elu", name="endconv3_1")
        conv3_2 = self.conv2d(input=conv3_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 64], activation="elu", name="endconv3_2")
        self.conv_shapes.append(conv3_2.shape)

        print("Encode-Dense")
        self.fc_shapes.append(conv3_2.shape)
        [n, h, w, c] = self.fc_shapes[0]
        fulcon_in = tf.compat.v1.reshape(conv3_2, shape=[self.batch_size, h*w*c], name="endfulcon_in")
        fulcon1 = self.fully_connected(input=fulcon_in, num_inputs=int(h*w*c), \
            num_outputs=512, activation="elu", name="endfullcon1")

        z_params = self.fully_connected(input=fulcon1, num_inputs=int(fulcon1.shape[1]), \
            num_outputs=self.z_dim*2, activation="None", name="endz_sigma")
        z_mu = z_params[:, :self.z_dim]
        z_sigma = z_params[:, self.z_dim:]

        z = self.sample_z(mu=z_mu, sigma=z_sigma) # reparameterization trick

        return z, z_mu, z_sigma

    def generator(self, input, ksize=3):

        print("\nGenerate-Dense")
        [n, h, w, c] = self.fc_shapes[0]
        fulcon2 = self.fully_connected(input=input, num_inputs=int(self.z_dim), \
            num_outputs=512, activation="elu", name="genfullcon2")
        fulcon3 = self.fully_connected(input=fulcon2, num_inputs=int(fulcon2.shape[1]), \
            num_outputs=int(h*w*c), activation="elu", name="genfullcon3")
        fulcon_out = tf.compat.v1.reshape(fulcon3, shape=[self.batch_size, h, w, c], name="genfulcon_out")

        print("Generate-1")
        convt1_1 = self.conv2d(input=fulcon_out, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 64], activation="elu", name="genconv1_1")
        convt1_2 = self.conv2d(input=convt1_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 64], activation="elu", name="genconv1_2")

        print("Generate-2")
        [n, h, w, c] = self.conv_shapes[-2]
        convt2_1 = self.conv2d_transpose(input=convt1_2, stride=2, padding='SAME', \
            output_shape=[self.batch_size, h, w, c], filter_size=[ksize, ksize, 32, 64], \
            dilations=[1, 1, 1, 1], activation="elu", name="genconv2_1")
        convt2_2 = self.conv2d(input=convt2_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 32], activation="elu", name="genconv2_2")

        print("Generate-3")
        [n, h, w, c] = self.conv_shapes[-3]
        convt3_1 = self.conv2d_transpose(input=convt2_2, stride=2, padding='SAME', \
            output_shape=[self.batch_size, h, w, c], filter_size=[ksize, ksize, 16, 32], \
            dilations=[1, 1, 1, 1], activation="elu", name="genconv3_1")
        convt3_2 = self.conv2d(input=convt3_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 16], activation="elu", name="genconv3_2")
        convt3_3 = self.conv2d(input=convt3_2, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 1], activation="sigmoid", name="genconv3_3")
        convt3_3 = tf.compat.v1.clip_by_value(convt3_3, 1e-12, 1-(1e-12))

        return convt3_3

    def transformer(self, input):

        print("\nTransformer-Dense")
        [n, m] = input.shape
        fulcon1 = self.fully_connected(input=input, num_inputs=m, \
            num_outputs=512, activation="elu", name="trafullcon1")
        fulcon2 = self.fully_connected(input=fulcon1, num_inputs=512, \
            num_outputs=m, activation="None", name="trafullcon2")

        return fulcon2

    def sample_z(self, mu, sigma):

        # default of tf.random.normal: mean=0.0, stddev=1.0
        epsilon = tf.random.normal(tf.shape(mu), dtype=tf.float32)
        sample = mu + (sigma * epsilon)

        return sample

    def initializer(self):
        return tf.compat.v1.initializers.variance_scaling(distribution="untruncated_normal", dtype=tf.dtypes.float32)

    def maxpool(self, input, ksize, strides, padding, name=""):

        out_maxp = tf.compat.v1.nn.max_pool(value=input, \
            ksize=ksize, strides=strides, padding=padding, name=name)
        print("Max-Pool", input.shape, "->", out_maxp.shape)

        return out_maxp

    def activation_fn(self, input, activation="relu", name=""):

        if("sigmoid" == activation):
            out = tf.compat.v1.nn.sigmoid(input, name='%s_sigmoid' %(name))
        elif("tanh" == activation):
            out = tf.compat.v1.nn.tanh(input, name='%s_tanh' %(name))
        elif("relu" == activation):
            out = tf.compat.v1.nn.relu(input, name='%s_relu' %(name))
        elif("lrelu" == activation):
            out = tf.compat.v1.nn.leaky_relu(input, name='%s_lrelu' %(name))
        elif("elu" == activation):
            out = tf.compat.v1.nn.elu(input, name='%s_elu' %(name))
        else: out = input

        return out

    def batch_normalization(self, input):

        mean = tf.compat.v1.reduce_mean(input)
        std = tf.compat.v1.math.reduce_std(input)

        return (input - mean) / (std + 1e-12)

    def conv2d(self, input, stride, padding, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], activation="relu", name=""):

        # strides=[N, H, W, C], [1, stride, stride, 1]
        # filter_size=[ksize, ksize, num_inputs, num_outputs]
        try:
            w_idx = self.w_names.index('%s_w' %(name))
            b_idx = self.b_names.index('%s_b' %(name))
        except:
            weight = tf.compat.v1.get_variable(name='%s_w' %(name), \
                shape=filter_size, initializer=self.initializer())
            bias = tf.compat.v1.get_variable(name='%s_b' %(name), \
                shape=[filter_size[-1]], initializer=self.initializer())

            self.weights.append(weight)
            self.biasis.append(bias)
            self.w_names.append('%s_w' %(name))
            self.b_names.append('%s_b' %(name))
        else:
            weight = self.weights[w_idx]
            bias = self.biasis[b_idx]

        out_conv = tf.compat.v1.nn.conv2d(
            input=input,
            filter=weight,
            strides=[1, stride, stride, 1],
            padding=padding,
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv' %(name),
        )
        out_bias = tf.math.add(out_conv, bias, name='%s_add' %(name))
        out_bias = self.batch_normalization(input=out_bias)

        print("Conv", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)

    def conv2d_transpose(self, input, stride, padding, output_shape, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], activation="relu", name=""):

        # strides=[N, H, W, C], [1, stride, stride, 1]
        # filter_size=[ksize, ksize, num_outputs, num_inputs]
        try:
            w_idx = self.w_names.index('%s_w' %(name))
            b_idx = self.b_names.index('%s_b' %(name))
        except:
            weight = tf.compat.v1.get_variable(name='%s_w' %(name), \
                shape=filter_size, initializer=self.initializer())
            bias = tf.compat.v1.get_variable(name='%s_b' %(name), \
                shape=[filter_size[-2]], initializer=self.initializer())

            self.weights.append(weight)
            self.biasis.append(bias)
            self.w_names.append('%s_w' %(name))
            self.b_names.append('%s_b' %(name))
        else:
            weight = self.weights[w_idx]
            bias = self.biasis[b_idx]

        out_conv = tf.compat.v1.nn.conv2d_transpose(
            value=input,
            filter=weight,
            output_shape=output_shape,
            strides=[1, stride, stride, 1],
            padding=padding,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv_tr' %(name),
        )
        out_bias = tf.math.add(out_conv, bias, name='%s_add' %(name))
        out_bias = self.batch_normalization(input=out_bias)

        print("Conv-Tr", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)

    def fully_connected(self, input, num_inputs, num_outputs, activation="relu", name=""):

        try:
            w_idx = self.w_names.index('%s_w' %(name))
            b_idx = self.b_names.index('%s_b' %(name))
        except:
            weight = tf.compat.v1.get_variable(name='%s_w' %(name), \
                shape=[num_inputs, num_outputs], initializer=self.initializer())
            bias = tf.compat.v1.get_variable(name='%s_b' %(name), \
                shape=[num_outputs], initializer=self.initializer())

            self.weights.append(weight)
            self.biasis.append(bias)
            self.w_names.append('%s_w' %(name))
            self.b_names.append('%s_b' %(name))
        else:
            weight = self.weights[w_idx]
            bias = self.biasis[b_idx]

        out_mul = tf.compat.v1.matmul(input, weight, name='%s_mul' %(name))
        out_bias = tf.math.add(out_mul, bias, name='%s_add' %(name))
        out_bias = self.batch_normalization(input=out_bias)

        print("Full-Con", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)
