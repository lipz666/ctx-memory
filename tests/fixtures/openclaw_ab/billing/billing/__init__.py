import logging
import sys

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s")
