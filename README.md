# Cursor Rules Pack

Cursor Composer(에이전트)의 작업 품질을 끌어올리기 위한 규칙 세트입니다.
탐색 → 계획 → 구현 → 검증 → 보고의 에이전트 워크플로우, 근본 원인 디버깅,
정직한 검증 보고 등 강력한 코딩 에이전트의 행동 패턴을 규칙으로 명문화했습니다.

## 설치

이 zip의 `.cursor/rules/` 폴더를 **프로젝트 루트**에 그대로 복사하세요:

```
your-project/
└── .cursor/
    └── rules/
        ├── 00-core-principles.mdc   (항상 적용)
        ├── 10-workflow.mdc          (항상 적용)
        ├── 20-code-quality.mdc      (항상 적용)
        ├── 30-debugging.mdc         (디버깅 상황에서 적용)
        ├── 40-testing.mdc           (테스트 관련 작업에 적용)
        ├── 50-typescript.mdc        (*.ts, *.tsx 등에서 자동 적용)
        ├── 55-react-frontend.mdc    (*.tsx, components/ 등에서 자동 적용)
        ├── 60-python.mdc            (*.py에서 자동 적용)
        ├── 65-backend-api.mdc       (api/, migrations/ 등에서 자동 적용)
        ├── 70-git-refactoring.mdc   (git/리팩터링 작업에 적용)
        └── 90-communication.mdc     (항상 적용)
```

- Cursor 최신 버전은 `.cursor/rules/*.mdc` 형식(frontmatter 포함)을 사용합니다.
- 구버전 Cursor 또는 단일 파일을 원하면, 파일들의 본문(frontmatter 제외)을 합쳐
  프로젝트 루트의 `.cursorrules` 파일 하나로 만들어도 됩니다.

## 커스터마이징 팁

- 항상 적용 규칙(alwaysApply: true)은 컨텍스트를 상시 차지하므로 3~4개 이하로 유지하는 게 좋습니다.
- 안 쓰는 언어 파일은 지워도 됩니다 (예: Python 안 쓰면 60-python.mdc 삭제).
- 프로젝트 고유 컨벤션(폴더 구조, 네이밍, 배포 절차)은 별도 파일
  (예: `05-project-context.mdc`, alwaysApply: true)로 추가하면 효과가 가장 큽니다.
  모델 성능 향상에 가장 크게 기여하는 것은 "이 프로젝트가 어떻게 생겼는지"에 대한 정보입니다.

## 참고

규칙 파일은 모델의 행동 습관(계획, 검증, 정직한 보고)을 교정해 실질적인 결과물 품질을
크게 개선하지만, 모델 자체의 추론 능력을 바꾸지는 못합니다. 규칙 + 좋은 프로젝트 컨텍스트
+ 작업을 작게 쪼개서 요청하는 습관을 합치면 가장 좋은 결과를 얻을 수 있습니다.
