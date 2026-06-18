"""
Celery task wrappers for video-processing functions.

Each task checks available RAM at startup. If less than 3 GB is free
it re-schedules itself 60 seconds later (up to 288 retries ≈ 4.8 hours)
rather than blocking a worker thread.

Start worker (concurrency capped at 7 via settings):
    celery -A Measure worker --loglevel=info
"""
import logging
import psutil
import redis
from celery import shared_task
from celery.exceptions import Retry
from django.conf import settings

from .task import (
    _process_sit_and_throw,
    _process_sit_and_reach,
    _process_video_task,
    _process_15m_dash,
    _process_plank,
    check_memory_available,
)

logger = logging.getLogger('homography_app')

_RETRY_COUNTDOWN = 60   # seconds to wait before re-checking memory
_MAX_RETRIES = 120      # 120 × 60 s = ~2 hours ceiling
_CONCURRENT_THRESHOLD = 4  # only enforce memory check when this many tasks are already running
_LOCK_RETRY_COUNTDOWN = 60  # seconds to wait before re-checking object lock
_LOCK_TIMEOUT = 3600        # auto-expire lock after 1 hour (safety net)

_redis = redis.Redis.from_url(settings.CELERY_BROKER_URL)


def enqueue_upload(petvideo_id, unique_id):
    """Push a unique_id to the per-object processing queue (called from views)."""
    _redis.rpush(f"petvideo_queue:{petvideo_id}", unique_id)
    queue_len = _redis.llen(f"petvideo_queue:{petvideo_id}")
    logger.info('Queue: video=%s uid=%s enqueued (position %d)', petvideo_id, unique_id, queue_len)


def _is_my_turn(petvideo_id, unique_id):
    """Return True only if this task is next in queue AND acquires the lock."""
    front = _redis.lindex(f"petvideo_queue:{petvideo_id}", 0)
    if front is not None and front.decode() != unique_id:
        logger.info('Queue: video=%s uid=%s waiting (front=%s)', petvideo_id, unique_id, front.decode())
        return False  # not my turn yet
    # Either I'm at front, or queue is empty — try to acquire lock
    acquired = bool(_redis.set(f"petvideo_lock:{petvideo_id}", "1", nx=True, ex=_LOCK_TIMEOUT))
    if acquired:
        logger.info('Queue: video=%s uid=%s acquired lock — processing', petvideo_id, unique_id)
    else:
        logger.info('Queue: video=%s uid=%s lock held by another task — retrying', petvideo_id, unique_id)
    return acquired


def _finish_turn(petvideo_id):
    """Pop the front of the queue and release the lock."""
    popped = _redis.lpop(f"petvideo_queue:{petvideo_id}")
    _redis.delete(f"petvideo_lock:{petvideo_id}")
    remaining = _redis.llen(f"petvideo_queue:{petvideo_id}")
    logger.info('Queue: video=%s uid=%s finished — %d remaining in queue',
                petvideo_id, popped.decode() if popped else 'none', remaining)


def _active_task_count():
    """Return total number of tasks currently executing across all workers."""
    try:
        from celery import current_app
        active = current_app.control.inspect(timeout=1).active() or {}
        return sum(len(tasks) for tasks in active.values())
    except Exception:
        return 0


def _check_or_retry(task):
    """If >= 3 concurrent tasks are running AND < 1.4 GB free, reschedule."""
    if _active_task_count() >= _CONCURRENT_THRESHOLD and not check_memory_available(min_gb=1.4):
        avail_gb = psutil.virtual_memory().available / 1024 ** 3
        logger.warning(
            'Task %s: only %.1f GB free with >=3 active tasks, rescheduling in %ds (attempt %d/%d)',
            task.name, avail_gb, _RETRY_COUNTDOWN,
            task.request.retries + 1, _MAX_RETRIES,
        )
        raise task.retry(countdown=_RETRY_COUNTDOWN, max_retries=_MAX_RETRIES)


@shared_task(bind=True, ignore_result=True, name='homography_app.celery_process_sit_and_throw')
def celery_process_sit_and_throw(self, petvideo_id, test_id='', assessment_id='', unique_id=''):
    if not _is_my_turn(petvideo_id, unique_id):
        raise self.retry(countdown=_LOCK_RETRY_COUNTDOWN, max_retries=_MAX_RETRIES)
    try:
        _check_or_retry(self)
        _process_sit_and_throw(petvideo_id, test_id=test_id, assessment_id=assessment_id, unique_id=unique_id)
    finally:
        _finish_turn(petvideo_id)


@shared_task(bind=True, ignore_result=True, name='homography_app.celery_process_sit_and_reach')
def celery_process_sit_and_reach(self, petvideo_id, test_id='', assessment_id='', unique_id=''):
    if not _is_my_turn(petvideo_id, unique_id):
        raise self.retry(countdown=_LOCK_RETRY_COUNTDOWN, max_retries=_MAX_RETRIES)
    try:
        _check_or_retry(self)
        _process_sit_and_reach(petvideo_id, test_id=test_id, assessment_id=assessment_id, unique_id=unique_id)
    finally:
        _finish_turn(petvideo_id)


@shared_task(bind=True, ignore_result=True, name='homography_app.celery_process_video_task')
def celery_process_video_task(
    self,
    petvideo_id,
    enable_color_marker_tracking=True,
    enable_start_end_detector=True,
    test_id='',
    assessment_id='',
    unique_id='',
):
    if not _is_my_turn(petvideo_id, unique_id):
        raise self.retry(countdown=_LOCK_RETRY_COUNTDOWN, max_retries=_MAX_RETRIES)
    try:
        _check_or_retry(self)
        _process_video_task(
            petvideo_id,
            enable_color_marker_tracking=enable_color_marker_tracking,
            enable_start_end_detector=enable_start_end_detector,
            test_id=test_id,
            assessment_id=assessment_id,
            unique_id=unique_id,
        )
    finally:
        _finish_turn(petvideo_id)


@shared_task(bind=True, ignore_result=True, name='homography_app.celery_process_15m_dash')
def celery_process_15m_dash(self, petvideo_id, test_id='', assessment_id='', unique_id=''):
    if not _is_my_turn(petvideo_id, unique_id):
        raise self.retry(countdown=_LOCK_RETRY_COUNTDOWN, max_retries=_MAX_RETRIES)
    try:
        _check_or_retry(self)
        _process_15m_dash(petvideo_id, test_id=test_id, assessment_id=assessment_id, unique_id=unique_id)
    finally:
        _finish_turn(petvideo_id)


@shared_task(bind=True, ignore_result=True, name='homography_app.celery_process_plank')
def celery_process_plank(self, petvideo_id, test_id='', assessment_id='', unique_id=''):
    if not _is_my_turn(petvideo_id, unique_id):
        raise self.retry(countdown=_LOCK_RETRY_COUNTDOWN, max_retries=_MAX_RETRIES)
    try:
        _check_or_retry(self)
        _process_plank(petvideo_id, test_id=test_id, assessment_id=assessment_id, unique_id=unique_id)
    finally:
        _finish_turn(petvideo_id)
