#!/usr/bin/env python3
"""Turn LMEE-Bench subset metadata into a leakage-separated CPU-only manifest.

This script DOES NOT run LMEE, produce StateMem observations, or compute QA accuracy.
The downloaded lmee_bench_sub JSONs are task/QA metadata, not temporal RGB logs.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path,
                   default=Path.home() / 'Downloads' / 'lmee_mini' / 'lmee_bench_sub',
                   help='Directory holding the downloaded LMEE subset JSON files')
    p.add_argument('--out-dir', type=Path,
                   default=Path.home() / 'Downloads' / 'lmee_adapter_out')
    p.add_argument('--max-tasks', type=int, default=None,
                   help='For a small smoke test; omit to index the entire downloaded subset')
    args = p.parse_args()

    root = args.root.expanduser().resolve()
    output = args.out_dir.expanduser().resolve()
    if not root.is_dir():
        p.error(f'Input directory does not exist: {root}')
    if args.max_tasks is not None and args.max_tasks < 1:
        p.error('--max-tasks must be positive')

    files = sorted(f for f in root.rglob('*.json') if '.cache' not in f.parts)
    if not files:
        p.error(f'No JSON metadata files found under {root}')

    tasks, queries, labels = [], [], []
    categories, difficulties = Counter(), Counter()
    scenes = set()
    missing_fields = Counter()
    total_qa = 0

    for source in files:
        with source.open(encoding='utf-8') as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            if 'tasks' in data and isinstance(data['tasks'], list):
                data = data['tasks']
            else:
                raise ValueError(f'Expected a task list in {source}')
        if not isinstance(data, list):
            raise ValueError(f'Expected a list in {source}, got {type(data).__name__}')

        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f'Non-object task in {source}, index {index}')
            task_id = f'{source.stem}:{index:04d}'
            scenes.add(item.get('Scene', 'UNKNOWN'))
            difficulty = item.get('Difficulty', 'UNKNOWN')
            difficulties[difficulty] += 1
            if not item.get('Task instruction'):
                missing_fields['Task instruction'] += 1

            # PUBLIC: supplied to the agent before exploration. No QA answer/path.
            tasks.append({
                'task_id': task_id,
                'scene': item.get('Scene'),
                'difficulty': difficulty,
                'instruction': item.get('Task instruction'),
                'target_object_ids': item.get('Object_id', []),
                'start_position': item.get('Start_position'),
                'start_rotation': item.get('Start_rotation'),
                'source_metadata_file': source.name,
            })

            for qa_index, qa in enumerate(item.get('QAs') or []):
                total_qa += 1
                qid = f'{task_id}:qa{qa_index:03d}'
                category = qa.get('category', 'UNKNOWN')
                categories[category] += 1
                if not qa.get('question'):
                    missing_fields['QA question'] += 1
                if not qa.get('answer'):
                    missing_fields['QA answer'] += 1

                # PUBLIC at question time. No ground-truth answer or gold frame.
                queries.append({
                    'question_id': qid,
                    'task_id': task_id,
                    'category': category,
                    'question': qa.get('question'),
                    'choices': qa.get('choices'),
                    'referenced_object_id': qa.get('object_id'),
                })
                # PRIVATE EVALUATION LABEL: never feed this to the agent or a
                # frame retriever. image_path points to an answer-supporting
                # reference frame and is therefore an oracle cue.
                labels.append({
                    'question_id': qid,
                    'task_id': task_id,
                    'answer': qa.get('answer'),
                    'open_answer': qa.get('open_answer'),
                    'gold_reference_image_path': qa.get('image_path'),
                })

            if args.max_tasks is not None and len(tasks) >= args.max_tasks:
                break
        if args.max_tasks is not None and len(tasks) >= args.max_tasks:
            break

    # Validate separation before writing anything.
    assert len({t['task_id'] for t in tasks}) == len(tasks), 'Duplicate task IDs'
    assert len({q['question_id'] for q in queries}) == len(queries), 'Duplicate QA IDs'
    assert len(queries) == len(labels) == total_qa
    assert all(not {'answer', 'open_answer', 'image_path', 'gold_reference_image_path'}
               .intersection(x) for x in tasks + queries), 'EVALUATION LEAKAGE'

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'tasks_public.jsonl', tasks)
    write_jsonl(output / 'queries_public.jsonl', queries)
    write_jsonl(output / 'labels_PRIVATE_eval_only.jsonl', labels)

    interface = {
        'purpose': 'Future actual RGB/log-derived inputs to the StateMem observation writer',
        'row_example_SCHEMA_ONLY': {
            'task_id': 'SCENE:0000', 'frame_index': 'integer >= 0',
            'entity_id': 'observed object ID', 'attribute': 'e.g. on_off / location',
            'observed_value': 'actual observed state',
            'raw_confidence': 'float [0,1], from perception not answer labels',
            'source_frame_path': 'actual observed frame path',
        },
        'important': [
            'The lmee_bench_sub JSONs do not contain observation timelines or RGB files.',
            'Never infer observations from QA answers, answer choices, or gold_reference_image_path.',
            'Do not fit reliability calibration on official test QA labels.',
            'This manifest alone cannot measure StateMem accuracy or compare it to LMEE.',
        ],
    }
    (output / 'observation_interface.json').write_text(
        json.dumps(interface, indent=2) + '\n', encoding='utf-8')

    summary = {
        'metadata_json_files_discovered': len(files),
        'tasks_indexed': len(tasks), 'questions_indexed': len(queries),
        'unique_scenes_in_indexed_tasks': len(scenes),
        'difficulty_counts': dict(sorted(difficulties.items())),
        'qa_category_counts': dict(sorted(categories.items())),
        'missing_field_counts': dict(sorted(missing_fields.items())),
        'rgb_frames_downloaded_by_this_script': 0,
        'has_observation_timeline_in_metadata': False,
        'official_benchmark_score': None,
    }
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print('LMEE MINI ADAPTER — metadata only; no GPU required')
    for k, v in summary.items():
        print(f'{k}: {v}')
    print(f'\nOutput: {output}')
    print('IMPORTANT: labels_PRIVATE_eval_only.jsonl and its gold image paths')
    print('must NEVER be read by the StateMem inference/retrieval pipeline.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
