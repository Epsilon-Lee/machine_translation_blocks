import logging
from machine_translation.model import BidirectionalEncoder, Decoder
from theano import tensor
from blocks.model import Model
from machine_translation.sampling import Bleuevaluator, Sampler
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
import theano.sandbox.cuda
import  argparse
from stream import get_dev_stream
import os
import pickle
logger = logging.getLogger(__name__)
from checkpoint import LoadData
# Use argparser to parse
parser = argparse.ArgumentParser()
parser.add_argument("--proto",  default="get_config_de2en",
                    help="Prototype config to use for config")
parser.add_argument("--gpu",  default="gpu0", help="choose a GPU to use")
parser.add_argument("--configuration_dir", default= '', help="Choose the dir where configurations file is placed.")
args = parser.parse_args()

if __name__ == "__main__":
    """
    We can't use mainloop and extensions anymore, it's better to directly write a function to evaluate
    """
    logger.info('Creating theano variables')
    #setting up GPU
    theano.sandbox.cuda.use(args.gpu)
    # Get configurations for model
    if args.configuration_dir:
        configurations_module = __import__(args.configuration_dir,fromlist=['configurations'])
        config = getattr(configurations_module.configurations, args.proto)()
    else:
        configurations_module = __import__('configurations')
        config = getattr(configurations_module, args.proto)()


    logger.info('Creating theano variables')
    source_sentence = tensor.lmatrix('source')
    sampling_input = tensor.lmatrix('input')

    # Construct model
    logger.info('Building RNN encoder-decoder')
    encoder = BidirectionalEncoder(
        config['src_vocab_size'], config['enc_embed'], config['enc_nhids'])
    decoder = Decoder(
        config['trg_vocab_size'], config['dec_embed'], config['dec_nhids'],
        config['enc_nhids'] * 2)
    logger.info("Building sampling model")
    # Sampling graph
    sampling_representation = encoder.apply(
        sampling_input, tensor.ones(sampling_input.shape))
    generated = decoder.generate(sampling_input, sampling_representation)
    search_model = Model(generated)
    # Filter the samples variable
    _, samples = VariableFilter(
        bricks=[decoder.sequence_generator], name="outputs")(
            ComputationGraph(generated[1]))  # generated[1] is next_outputs
    # Data directory
    datadir = 'de-en-data'
    test_stream = get_dev_stream(val_set= os.path.join(datadir,'test.de.small.tok'),
                                 src_vocab=os.path.join(datadir,'vocab.de-en.de.pkl'),
                                 src_vocab_size=30000)
    load_tedtalk = LoadData(saveto=config['saveto'])
    load_tedtalk.load_to(search_model)
    with open(config['trg_vocab']) as f:
        trg_vocab = pickle.load(f)
    evaluator = Bleuevaluator(source_sentence=sampling_input,
                              data_stream=test_stream,
                              config=config,
                              samples=samples,
                              search_model_de2en=search_model,
                              trg_vocab=trg_vocab,
                              val_set_grndtruth=os.path.join(datadir,'test.en'))
    evaluator.evaluate_model()


