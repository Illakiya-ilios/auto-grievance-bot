import { useState } from 'react'
import { ArrowDownLeft, ArrowUpRight, Check, ChevronDown, CircleHelp, FilePlus2, LifeBuoy, LoaderCircle, LockKeyhole, MessageSquareText, Pencil, Plus, Send, ShieldCheck, X } from 'lucide-react'
import { callApi } from './api.js'
const FIELD_LABELS = {
  name: 'Full name',
  contact: 'Phone or email',
  recipient: 'Person or department to receive grievance',
  category: 'Category',
  subject: 'Subject',
  description: 'What happened',
  related_date: 'Related date',
  supporting_information: 'Supporting information',
}
const REQUIRED = ['name', 'contact', 'recipient', 'category', 'subject', 'description']

function App() {
  const [conversationId, setConversationId] = useState(() => sessionStorage.getItem('conversationId'))
  const [messages, setMessages] = useState([])
  const [draft, setDraft] = useState('')
  const [review, setReview] = useState(null)
  const [editing, setEditing] = useState(false)
  const [editDetails, setEditDetails] = useState({})
  const [grievance, setGrievance] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [activeAction, setActiveAction] = useState('')

  function addMessage(role, text, extra = {}) {
    setMessages((current) => [...current, { role, text, ...extra }])
  }

  async function sendMessage(value = draft) {
    const message = value.trim()
    if (!message || busy) return
    setDraft('')
    setError('')
    addMessage('user', message)
    setBusy(true)
    try {
      const result = await callApi('/chat', {
        method: 'POST',
        body: JSON.stringify({
          message,
          conversation_id: conversationId,
        }),
      })
      setConversationId(result.conversation_id)
      sessionStorage.setItem('conversationId', result.conversation_id)
      addMessage('assistant', result.message)
      if (result.review) {
        setReview(result.review)
        setEditDetails(result.review)
        setEditing(false)
      }
      if (result.grievance) setGrievance(result.grievance)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setBusy(false)
    }
  }

  async function submitDecision(action, details) {
    if (!conversationId || busy) return
    setBusy(true)
    setError('')
    try {
      const result = await callApi(`/conversations/${conversationId}/decision`, {
        method: 'POST',
        body: JSON.stringify({ action, details }),
      })
      if (action === 'edit') {
        setReview(result.review)
        setEditDetails(result.review)
        setEditing(false)
      } else {
        setReview(null)
        setEditing(false)
        if (result.grievance) setGrievance(result.grievance)
        if (action === 'approve') {
          sessionStorage.removeItem('conversationId')
          setConversationId(null)
        }
        addMessage('assistant', result.message)
      }
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setBusy(false)
    }
  }

  function startFresh() {
    sessionStorage.removeItem('conversationId')
    setConversationId(null)
    setMessages([])
    setReview(null)
    setGrievance(null)
    setError('')
    setActiveAction('')
  }

  function quickAction(action) {
    setActiveAction(action)
    const prompts = {
      register: 'I want to register a grievance.',
      track: 'I want to track a grievance.',
      escalate: 'I need to escalate an unresolved grievance.',
    }
    sendMessage(prompts[action])
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark"><MessageSquareText size={19} strokeWidth={2.2} /></div>
          <div><span className="brand-name">Civicline</span><span className="brand-caption">PUBLIC SERVICE DESK</span></div>
        </div>
        <div className="side-rule" />
        <div className="nav-label">YOUR DESK</div>
        <button className={`nav-item ${activeAction === 'register' ? 'selected' : ''}`} onClick={() => quickAction('register')}><FilePlus2 size={17} />Register a grievance</button>
        <button className={`nav-item ${activeAction === 'track' ? 'selected' : ''}`} onClick={() => quickAction('track')}><ArrowDownLeft size={17} />Track a request</button>
        <button className={`nav-item ${activeAction === 'escalate' ? 'selected' : ''}`} onClick={() => quickAction('escalate')}><ArrowUpRight size={17} />Escalate an issue</button>
        <div className="side-note">
          <span className="side-note-icon"><ShieldCheck size={16} /></span>
          <div><strong>Your information, handled carefully.</strong><p>A grievance is only submitted after you review and approve it.</p></div>
        </div>
        <div className="sidebar-bottom">
          <div className="agent-status"><span className="status-dot" />Assistant available</div>
          <div className="help-link"><CircleHelp size={15} />Need help? Start a message</div>
          <div className="desk-version">CIVICLINE · SERVICE DESK</div>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div className="breadcrumb"><span>SUPPORT</span><ChevronDown size={13} /><span className="breadcrumb-current">CONVERSATION</span></div>
          <div className="topbar-right"><span className="secure-label"><LockKeyhole size={13} />Private conversation</span><button className="icon-button new-chat" title="Start a new conversation" aria-label="Start a new conversation" onClick={startFresh}><Plus size={18} /></button></div>
        </header>

        <section className="conversation-wrap">
          <div className="conversation-heading">
            <div>
              <div className="section-kicker"><span className="live-dot" />SERVICE DESK <span className="kicker-divider">/</span> ONLINE</div>
              <h1>Tell me what’s happening.</h1>
              <p className="heading-subtitle">Describe the issue covered by your service provider’s reference documents. I’ll ask follow-up questions before preparing a grievance ticket.</p>
            </div>
            <div className="reference-chip"><span>CASE SUPPORT</span><span className="chip-line" /><LifeBuoy size={17} /></div>
          </div>

          <div className="chat-surface">
            <div className="chat-log" aria-live="polite">
              {messages.map((message, index) => (
                <div className={`message-row ${message.role}`} key={`${index}-${message.text.slice(0, 12)}`}>
                  {message.role === 'assistant' && <div className="assistant-avatar"><MessageSquareText size={15} /></div>}
                  <div className="message-content">
                    {message.role === 'assistant' && <div className="speaker-label">CIVICLINE <span>·</span> SUPPORT</div>}
                    <div className="message-bubble">{message.text}</div>
                  </div>
                </div>
              ))}

              {review && <ReviewPanel review={review} editing={editing} values={editDetails} busy={busy} onEdit={() => { setEditing(true); setEditDetails(review) }} onDiscard={() => { setEditing(false); setEditDetails(review) }} onChange={(field, value) => setEditDetails((current) => ({ ...current, [field]: value }))} onSave={() => submitDecision('edit', editDetails)} onApprove={() => submitDecision('approve')} onCancel={() => submitDecision('cancel')} />}
              {grievance && <StatusPanel grievance={grievance} />}
              {busy && <div className="working-indicator"><LoaderCircle size={16} className="spin" />Preparing your response</div>}
              {error && <div className="error-banner"><CircleHelp size={15} />{error}</div>}
            </div>

            <form className="composer" onSubmit={(event) => { event.preventDefault(); sendMessage() }}>
              <label className="sr-only" htmlFor="message-input">Your message</label>
              <input id="message-input" value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Write a message..." disabled={busy} maxLength={4000} />
              <span className="composer-hint">Press Enter to send</span>
              <button type="submit" className="send-button" aria-label="Send message" disabled={!draft.trim() || busy}><Send size={17} /></button>
            </form>
          </div>
          <footer className="conversation-footer"><span><LockKeyhole size={12} />Your details are only used to handle your request.</span><span>For urgent matters, contact your service provider directly.</span></footer>
        </section>
      </main>
    </div>
  )
}

function ReviewPanel({ review, editing, values, busy, onEdit, onDiscard, onChange, onSave, onApprove, onCancel }) {
  const fields = Object.keys(FIELD_LABELS).filter((field) => values[field] || review[field])
  return (
    <div className="review-panel">
      <div className="review-head"><div><div className="review-eyebrow"><span className="review-dot" />DRAFT · NOT YET SUBMITTED</div><h2>Review your grievance</h2></div><div className="review-stamp"><ShieldCheck size={17} />YOUR APPROVAL REQUIRED</div></div>
      <p className="review-intro">Check that everything looks right. Your grievance is not registered until you approve it.</p>
      <div className="review-fields">
        {fields.map((field) => (
          <label className={`review-field ${field === 'description' || field === 'supporting_information' ? 'wide' : ''}`} key={field}>
            <span>{FIELD_LABELS[field]}{REQUIRED.includes(field) && <i>Required</i>}</span>
            {editing ? field === 'description' || field === 'supporting_information' ? <textarea value={values[field] || ''} onChange={(event) => onChange(field, event.target.value)} rows={field === 'description' ? 3 : 2} /> : <input value={values[field] || ''} onChange={(event) => onChange(field, event.target.value)} /> : <strong>{review[field]}</strong>}
          </label>
        ))}
      </div>
      <div className="review-actions">
        {editing ? <><button className="button-muted" onClick={onDiscard} type="button" disabled={busy}><X size={15} />Discard edits</button><button className="button-primary" onClick={onSave} type="button" disabled={busy}><Check size={15} />Save changes</button></> : <><button className="button-muted" onClick={onEdit} type="button" disabled={busy}><Pencil size={15} />Edit details</button><button className="button-muted cancel-button" onClick={onCancel} type="button" disabled={busy}><X size={15} />Cancel</button><button className="button-primary" onClick={onApprove} type="button" disabled={busy}><Check size={16} />Approve &amp; submit</button></>}
      </div>
    </div>
  )
}

function StatusPanel({ grievance }) {
  return (
    <div className="status-panel">
      <div className="status-panel-head"><div><span className="status-eyebrow">GRIEVANCE RECEIPT</span><h2>{grievance.grievance_id}</h2></div><span className={`status-pill ${grievance.status}`}>{grievance.status}</span></div>
      <div className="status-subject">{grievance.subject || grievance.description}</div>
      {grievance.category && <div className="status-subject">Category: <strong>{grievance.category}</strong></div>}
      {grievance.recipient && <div className="status-subject">Send this receipt to: <strong>{grievance.recipient}</strong></div>}
      {grievance.description && <p className="review-intro">{grievance.description}</p>}
      {grievance.supporting_information && <div className="status-subject status-details"><strong>Additional information</strong><br />{grievance.supporting_information}</div>}
      {grievance.history?.length > 0 && <div className="status-history"><span className="history-mark" /><div><strong>{grievance.history[grievance.history.length - 1].status}</strong><span>{grievance.history[grievance.history.length - 1].note}</span></div></div>}
    </div>
  )
}

export default App
