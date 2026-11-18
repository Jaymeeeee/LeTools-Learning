"""按官方轨迹元数据批量切分头部视频，只输出review/episode编号.mp4。"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import accumulate
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import pyarrow.parquet as pq

SOURCE = Path('/root/gpufree-data/openlet/dataset-task2-original')
CAMERA = 'videos/observation.images.head_cam_h/'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--workers', type=int, choices=range(1, 13), default=2)
    args = parser.parse_args()
    source = args.source.resolve()
    fps = json.loads((source / 'meta/info.json').read_text())['fps']
    columns = ['episode_index', 'length'] + [CAMERA + name for name in
               ('chunk_index', 'file_index', 'from_timestamp', 'to_timestamp')]
    bundles = defaultdict(list)
    for path in sorted((source / 'meta/episodes').rglob('*.parquet')):
        for row in pq.read_table(path, columns=columns).to_pylist():
            video = source / 'videos/observation.images.head_cam_h' / (
                f'chunk-{row[CAMERA + "chunk_index"]:03d}/file-{row[CAMERA + "file_index"]:03d}.mp4')
            bundles[video].append(row)

    output = source / 'review'
    output.mkdir(exist_ok=True)
    if any(not p.is_file() or p.suffix != '.mp4' for p in output.iterdir()):
        raise ValueError(f'请先移走review里的旧目录及非MP4文件：{output}')
    help_text = subprocess.run(['ffmpeg', '-hide_banner', '-h', 'full'],
                               capture_output=True, text=True, check=True).stdout
    sync = ['-fps_mode', 'passthrough'] if '-fps_mode' in help_text else ['-vsync', '0']

    def split(item):
        video, rows = item
        rows.sort(key=lambda row: row[CAMERA + 'from_timestamp'])
        ends = list(accumulate(row['length'] for row in rows))
        for row, end in zip(rows, ends):
            if (abs(row[CAMERA + 'from_timestamp'] * fps - (end - row['length'])) > 1e-4
                    or abs(row[CAMERA + 'to_timestamp'] * fps - end) > 1e-4):
                raise ValueError(f'源视频轨迹不连续，不能按累计帧数切分：{video}')
        with TemporaryDirectory(prefix='.split-', dir=output) as temp:
            command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                       '-threads', '2', '-i', str(video), '-map', '0:v:0', '-an',
                       '-vf', 'setpts=PTS-STARTPTS', *sync, '-frames:v', str(ends[-1]),
                       '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                       '-threads', '2', '-bf', '0', '-sc_threshold', '0']
            if len(rows) > 1:
                command += ['-force_key_frames', ','.join(f'{n / fps:.6f}' for n in ends[:-1])]
            command += ['-f', 'segment', '-segment_format', 'mp4', '-reset_timestamps', '1',
                        '-segment_format_options', 'movflags=+faststart']
            command += (['-segment_frames', ','.join(map(str, ends[:-1]))] if len(rows) > 1
                        else ['-segment_time', '100000000'])
            subprocess.run(command + [str(Path(temp) / 'part-%04d.mp4')], check=True)
            clips = sorted(Path(temp).glob('part-*.mp4'))
            if len(clips) != len(rows):
                raise RuntimeError(f'切分数量与轨迹数不一致：{video}')
            for clip, row in zip(clips, rows):
                clip.replace(output / f'episode{row["episode_index"]:03d}.mp4')
        print(f'{video.name}: {len(rows)}条头部视频', flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(split, bundles.items()))
    print(f'完成：{sum(map(len, bundles.values()))}条头部视频，输出到 {output}')

if __name__ == '__main__':
    main()