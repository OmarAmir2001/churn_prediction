import logging

import structlog


log = structlog.get_logger()

log.debug("this_is_hidden", detail="too much detail")
log.info("training_started", rows=7043, C=1.0)