

(function() {

	var Graph = function() {};
	Graph.prototype = {
		add: function(step, y) {
			console.log(step, y);
		},
		drawSelf: function(canv) {}
	};

	var lossGraph = new Graph();
	var xLossWindow = new cnnutil.Window(100);
	var wLossWindow = new cnnutil.Window(100);
	var trainAccWindow = new cnnutil.Window(100);
	var valAccWindow = new cnnutil.Window(100);
	var testAccWindow = new cnnutil.Window(50, 1);
	var step_num = 0;

	var img_data = new Array(num_batches);
	var loaded = new Array(num_batches);
	var loaded_train_batches = [];
	
	var classes_txt = ['0','1','2','3','4','5','6','7','8','9'];
	var dataset_name = "mnist";
	var num_batches = 21;
	var test_batch = 20;
	var num_samples_per_batch = 3000;
	var image_dimension = 28;
	var image_channels = 1;
	var use_validation_data = true;
	var random_flip = false;
	var random_position = false;
	var layer_defs,
		net,
		trainer;
	var trainer_text = "layer_defs = [];\n\
			layer_defs.push({type:'input', out_sx:24, out_sy:24, out_depth:1});\n\
			layer_defs.push({type:'conv', sx:5, filters:8, stride:1, pad:2, activation:'relu'});\n\
			layer_defs.push({type:'pool', sx:2, stride:2});\n\
			layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});\n\
			layer_defs.push({type:'pool', sx:3, stride:3});\n\
			layer_defs.push({type:'softmax', num_classes:10});\n\
			\n\
			net = new convnetjs.Net();\n\
			net.makeLayers(layer_defs);\n\
			\n\
			trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:20, l2_decay:0.001});\n\
			";

	var img_data = new Array(num_batches),
		loaded   = new Array(num_batches),
		loaded_train_batches = [];



	// mnist object
	var mnist = {
		isPaused: false,
		init: function() {

			eval(trainer_text);

			for (var k=0; k<loaded.length; k++) {
				loaded[k] = false;
			}

			// async load train set batch 0
			mnist.load_data_batch(0, function() {
				// async load test set
				mnist.load_data_batch(test_batch, function() {
					mnist.interval = setInterval(mnist.load_and_step, 0);
				});
			});
		},
		toggle_pause: function() {
			mnist.isPaused = !mnist.isPaused;

			if (mnist.isPaused) {
				clearInterval(mnist.interval);
			} else {
				mnist.interval = setInterval(mnist.load_and_step, 0);
			}
		},
		load_and_step: function() {
			if (mnist.isPaused) return;
			var sample = mnist.sample_training_instance();
			mnist.step(sample); // process this image
		},
		load_data_batch: function(batch_num, callback) {
			// Load the dataset with JS in background
			var img = new Image(),
				data_canvas = document.createElement('canvas'),
				data_ctx = data_canvas.getContext("2d");

			img.onload = function() {
				data_canvas.width = this.width;
				data_canvas.height = this.height;

				data_ctx.drawImage(this, 0, 0);

				img_data[batch_num] = data_ctx.getImageData(0, 0, data_canvas.width, data_canvas.height);
				loaded[batch_num] = true;

				if (batch_num < test_batch) {
					loaded_train_batches.push(batch_num);
				}
				console.log('finished loading data batch '+ batch_num);

				if (typeof(callback) === 'function') {
					callback();
				}
			};

			img.src = dataset_name +"/"+ dataset_name +"_batch_"+ batch_num +".png";
		},
		test_predict: function() {
			var num_classes = net.layers[net.layers.length-1].out_depth;

			var num_total = 0;
			var num_correct = 0;

			// grab a random test image
			for (num=0;num<4;num++) {
				var sample = mnist.sample_test_instance();
				var y = sample.label;  // ground truth label

				// forward prop it through the network
				var aavg = new convnetjs.Vol(1,1,num_classes,0.0);
				// ensures we always have a list, regardless if above returns single item or list
				var xs = [].concat(sample.x);
				var n = xs.length;
				for (var i=0;i<n;i++) {
					var a = net.forward(xs[i]);
					aavg.addFrom(a);
				}
				var preds = [];
				for (var k=0; k<aavg.w.length; k++) {
					preds.push({k:k, p:aavg.w[k]});
				}
				preds.sort(function(a,b) {return a.p<b.p ? 1:-1;});
				
				var correct = preds[0].k===y;
				if (correct) num_correct++;
				num_total++;
			}
			testAccWindow.add(num_correct/num_total);
			console.log('test accuracy based on last 200 test images: ' + testAccWindow.get_average());  
		},
		sample_training_instance: function() {
			// find an unloaded batch
			var bi = Math.floor(Math.random() * loaded_train_batches.length),
				b = loaded_train_batches[bi],
				k = Math.floor(Math.random() * num_samples_per_batch), // sample within the batch
				n = b*num_samples_per_batch + k;

			// load more batches over time
			if (step_num % (2 * num_samples_per_batch) === 0 && step_num > 0) {
				for (var i=0; i<num_batches; i++) {
					if (!loaded[i]) {
						// load it
						mnist.load_data_batch(i);
						break; // okay for now
					}
				}
			}

			// fetch the appropriate row of the training image and reshape into a Vol
			var p = img_data[b].data,
				x = new convnetjs.Vol(image_dimension,image_dimension,image_channels,0.0),
				W = image_dimension*image_dimension,
				j=0;

			for (var dc=0; dc<image_channels; dc++) {
				var i = 0;
				for (var xc=0; xc<image_dimension; xc++) {
					for (var yc=0; yc<image_dimension; yc++) {
						var ix = ((W * k) + i) * 4 + dc;
						x.set(yc, xc, dc, p[ix] / 255.0 - 0.5);
						i++;
					}
				}
			}

			if (random_position) {
				var dx = Math.floor(Math.random() * 5-2),
					dy = Math.floor(Math.random() * 5-2);
				x = convnetjs.augment(x, image_dimension, dx, dy, false); //maybe change position
			}

			if (random_flip) {
				x = convnetjs.augment(x, image_dimension, 0, 0, Math.random() < 0.5); //maybe flip horizontally
			}

			var isval = use_validation_data && n % 10 === 0 ? true : false;
			return {x: x, label: labels[n], isval: isval};
		},
		// evaluate current network on test set
		step: function(sample) {
			var x = sample.x;
			var y = sample.label;

			if (sample.isval) {
				// use x to build our estimate of validation error
				net.forward(x);
				var yhat = net.getPrediction();
				var val_acc = yhat === y ? 1.0 : 0.0;
				valAccWindow.add(val_acc);
				return; // get out
			}

			// train on it with network
			var stats = trainer.train(x, y);
			var lossx = stats.cost_loss;
			var lossw = stats.l2_decay_loss;

			// keep track of stats such as the average training error and loss
			var yhat = net.getPrediction();
			var train_acc = yhat === y ? 1.0 : 0.0;
			xLossWindow.add(lossx);
			wLossWindow.add(lossw);
			trainAccWindow.add(train_acc);

			// log progress to graph, (full loss)
			if (step_num % 200 === 0) {
				var xa = xLossWindow.get_average();
				var xw = wLossWindow.get_average();
				if (xa >= 0 && xw >= 0) { // if they are -1 it means not enough data was accumulated yet for estimates
					lossGraph.add(step_num, xa + xw);
					lossGraph.drawSelf(document.getElementById("lossgraph"));
				}
			}

			// run prediction on test set
			if ((step_num % 100 === 0 && step_num > 0) || step_num===100) {
				mnist.test_predict();
			}
			step_num++;
		},
		// sample a random testing instance
		sample_test_instance: function() {
			var b = test_batch,
				k = Math.floor(Math.random() * num_samples_per_batch),
				n = b * num_samples_per_batch + k;

			var p = img_data[b].data,
				x = new convnetjs.Vol(image_dimension, image_dimension, image_channels, 0.0),
				W = image_dimension * image_dimension,
				j=0;
			for (var dc=0; dc<image_channels; dc++) {
				var i=0;
				for (var xc=0; xc<image_dimension; xc++) {
					for (var yc=0; yc<image_dimension; yc++) {
						var ix = ((W * k) + i) * 4 + dc;
						x.set(yc, xc, dc, p[ix] / 255.0 - 0.5);
						i++;
					}
				}
			}
			// distort position and maybe flip
			var xs = [];
			
			if (random_flip || random_position) {
				for (var k=0; k<6; k++) {
					var test_variation = x;
					if (random_position) {
						var dx = Math.floor(Math.random() * 5-2);
						var dy = Math.floor(Math.random() * 5-2);
						test_variation = convnetjs.augment(test_variation, image_dimension, dx, dy, false);
					}
					
					if (random_flip) {
						test_variation = convnetjs.augment(test_variation, image_dimension, 0, 0, Math.random()<0.5); 
					}

					xs.push(test_variation);
				}
			} else {
				xs.push(x, image_dimension, 0, 0, false); // push an un-augmented copy
			}
			// return multiple augmentations, and we will average the network over them
			// to increase performance
			return {x: xs, label: labels[n]};
		}
	};

	window.mnist = mnist;

	mnist.init();


})();

