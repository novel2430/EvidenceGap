import { useState, type FormEvent } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  AlertTriangle,
  Check,
  Languages,
  LoaderCircle,
  Plus,
  RefreshCw,
  X,
} from 'lucide-react'
import { evidenceGapApi } from '../api'
import type { LocalizationStatusResponse } from '../contracts'
import { useToast } from '../hooks/useToast'
import { UI_TEXT } from '../uiText'
import {
  formatCreatedAt,
  getApiErrorMessage,
  getToastErrorMessage,
} from '../utils/format'

interface LocalizationPanelProps {
  runId: string
  originalLanguage: string
  localizations: LocalizationStatusResponse[]
  selectedLocalizationId: string | null
  activeLocalizationId: string | null
  selectedLocalization: LocalizationStatusResponse | null
  isListLoading: boolean
  listErrorMessage: string | null
  selectedLoadErrorMessage: string | null
  selectionNotice: string | null
  onRetryList: () => void
  onRetrySelected: () => void
  onSelectLocalization: (localizationId: string | null) => void
}

function LocalizationStatusIcon({
  status,
}: {
  status: LocalizationStatusResponse['status']
}) {
  if (status === 'succeeded') return <Check size={12} />
  if (status === 'failed') return <X size={12} />
  return <LoaderCircle className="spin" size={12} />
}

export function LocalizationPanel({
  runId,
  originalLanguage,
  localizations,
  selectedLocalizationId,
  activeLocalizationId,
  selectedLocalization,
  isListLoading,
  listErrorMessage,
  selectedLoadErrorMessage,
  selectionNotice,
  onRetryList,
  onRetrySelected,
  onSelectLocalization,
}: LocalizationPanelProps) {
  const [targetLanguage, setTargetLanguage] = useState<string>(
    UI_TEXT.localization.commonLanguages[0],
  )
  const [validationError, setValidationError] = useState<string | null>(null)
  const queryClient = useQueryClient()
  const { showToast, trackOperation } = useToast()
  const createToastId = `localization-create:${runId}`
  const createLocalization = useMutation({
    mutationFn: (language: string) =>
      evidenceGapApi.createLocalization(runId, { language }),
    onMutate: (language) => {
      showToast({
        id: createToastId,
        type: 'loading',
        title: UI_TEXT.toast.localizationCreating,
        description: UI_TEXT.toast.localizationRequesting(language),
      })
    },
    onSuccess: (created) => {
      trackOperation(
        `localization:${created.source_run_id}:${created.localization_id}`,
      )
      showToast({
        id: createToastId,
        type: 'info',
        title: UI_TEXT.toast.localizationStarted,
        description: UI_TEXT.toast.localizationGenerating(created.language),
      })
      void queryClient.invalidateQueries({ queryKey: ['localizations', runId] })
      onSelectLocalization(created.localization_id)
    },
    onError: (error) => {
      showToast({
        id: createToastId,
        type: 'error',
        title: UI_TEXT.toast.localizationStartFailed,
        description: getToastErrorMessage(error),
      })
    },
  })

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const language = targetLanguage.trim()
    if (!language) {
      setValidationError(UI_TEXT.localization.validation)
      return
    }
    setValidationError(null)
    createLocalization.mutate(language)
  }

  const createError = createLocalization.isError
    ? getApiErrorMessage(createLocalization.error)
    : validationError

  return (
    <section className="localization-panel">
      <div className="progress-heading">
        <div>
          <span className="eyebrow">{UI_TEXT.localization.eyebrow}</span>
          <h2>{UI_TEXT.localization.title}</h2>
        </div>
        <Languages size={17} />
      </div>

      <div className="localization-list">
        <button
          className={[
            'localization-item',
            selectedLocalizationId === null ? 'is-selected' : '',
            activeLocalizationId === null ? 'is-active-presentation' : '',
          ].filter(Boolean).join(' ')}
          type="button"
          onClick={() => onSelectLocalization(null)}
        >
          <span className="localization-item-main">
            <strong>{UI_TEXT.common.original}</strong>
            <small>{originalLanguage}</small>
          </span>
          <span className="localization-status localization-status--succeeded">
            <Check size={12} /> {UI_TEXT.localization.source}
          </span>
        </button>

        {isListLoading && (
          <div className="localization-list-state" role="status">
            <LoaderCircle className="spin" size={14} />
            {UI_TEXT.localization.loading}
          </div>
        )}

        {listErrorMessage && (
          <div className="localization-error" role="alert">
            <AlertTriangle size={14} />
            <div>
              <strong>{UI_TEXT.localization.listLoadFailed}</strong>
              <p>{listErrorMessage}</p>
              <button type="button" onClick={onRetryList}>
                <RefreshCw size={12} /> {UI_TEXT.common.retry}
              </button>
            </div>
          </div>
        )}

        {!isListLoading && !listErrorMessage && localizations.length === 0 && (
          <p className="localization-empty">{UI_TEXT.localization.empty}</p>
        )}

        {localizations.map((listedLocalization) => {
          const localization =
            selectedLocalization?.localization_id ===
            listedLocalization.localization_id
              ? selectedLocalization
              : listedLocalization
          return (
            <button
              className={[
                'localization-item',
                selectedLocalizationId === localization.localization_id
                  ? 'is-selected'
                  : '',
                activeLocalizationId === localization.localization_id
                  ? 'is-active-presentation'
                  : '',
              ].filter(Boolean).join(' ')}
              type="button"
              key={localization.localization_id}
              onClick={() =>
                onSelectLocalization(localization.localization_id)}
            >
              <span className="localization-item-main">
                <strong>{localization.language}</strong>
                <small>{formatCreatedAt(localization.created_at)}</small>
              </span>
              <span className={`localization-status localization-status--${localization.status}`}>
                <LocalizationStatusIcon status={localization.status} />
                {UI_TEXT.statusLabels[localization.status]}
              </span>
            </button>
          )
        })}
      </div>

      {selectedLocalization &&
        (selectedLocalization.status === 'queued' ||
          selectedLocalization.status === 'running') && (
          <div className="localization-pending" role="status">
            <LoaderCircle className="spin" size={14} />
            <span>
              <strong>{selectedLocalization.language}</strong>
              {selectedLocalization.status === 'queued'
                ? UI_TEXT.localization.queued
                : UI_TEXT.localization.running}
            </span>
          </div>
        )}

      {selectedLocalization?.status === 'failed' && (
        <div className="localization-error" role="alert">
          <AlertTriangle size={14} />
          <div>
            <strong>
              {selectedLocalization.error?.code ??
                UI_TEXT.localization.failedFallback}
            </strong>
            <p>
              {selectedLocalization.error?.message ??
                UI_TEXT.localization.missingError}
            </p>
            <small>{UI_TEXT.localization.retryDescription}</small>
          </div>
        </div>
      )}

      {(selectedLoadErrorMessage || selectionNotice) && (
        <div className="localization-error" role="alert">
          <AlertTriangle size={14} />
          <div>
            <strong>{UI_TEXT.localization.originalDisplayed}</strong>
            <p>{selectedLoadErrorMessage ?? selectionNotice}</p>
            {selectedLoadErrorMessage && (
              <button type="button" onClick={onRetrySelected}>
                <RefreshCw size={12} /> {UI_TEXT.common.retry}
              </button>
            )}
          </div>
        </div>
      )}

      <form className="localization-form" onSubmit={handleSubmit}>
        <label htmlFor="target-language">
          {UI_TEXT.localization.targetLabel}
        </label>
        <input
          id="target-language"
          list="target-language-options"
          value={targetLanguage}
          onChange={(event) => setTargetLanguage(event.target.value)}
          disabled={createLocalization.isPending}
          maxLength={100}
        />
        <datalist id="target-language-options">
          {UI_TEXT.localization.commonLanguages.map((language) => (
            <option value={language} key={language} />
          ))}
        </datalist>
        <button
          className="secondary-button"
          type="submit"
          disabled={createLocalization.isPending}
        >
          {createLocalization.isPending ? (
            <LoaderCircle className="spin" size={13} />
          ) : (
            <Plus size={13} />
          )}
          {createLocalization.isPending
            ? UI_TEXT.localization.creating
            : UI_TEXT.localization.create}
        </button>
        {createError && <p className="form-error" role="alert">{createError}</p>}
      </form>
    </section>
  )
}
