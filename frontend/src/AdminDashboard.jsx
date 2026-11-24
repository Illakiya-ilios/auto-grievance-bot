import { useEffect, useState } from 'react'
import { CircleHelp, FilePlus2, FileText, ShieldCheck, Trash2 } from 'lucide-react'
import { callApi } from './api.js'

export default function AdminDashboard() {
  const [token, setToken] = useState(() => sessionStorage.getItem('adminToken') || '')
  const [password, setPassword] = useState('')
  const [documents, setDocuments] = useState([])
  const [selectedFile, setSelectedFile] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  useEffect(() => {
    if (token) loadDocuments(token)
  }, [token])

  function authOptions(method) {
    return {
      method,
      headers: { Authorization: `Bearer ${token}` },
    }
  }

  async function loadDocuments(adminToken = token) {
    try {
      const result = await callApi('/admin/documents', {
        headers: { Authorization: `Bearer ${adminToken}` },
      })
      setDocuments(result.documents)
    } catch (requestError) {
      setError(requestError.message)
      if (requestError.message.toLowerCase().includes('session')) {
        sessionStorage.removeItem('adminToken')
        setToken('')
      }
    }
  }

  async function login(event) {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      const result = await callApi('/admin/login', {
        method: 'POST',
        body: JSON.stringify({ password }),
      })
      sessionStorage.setItem('adminToken', result.token)
      setToken(result.token)
      setPassword('')
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setBusy(false)
    }
  }

  async function uploadDocument(event) {
    event.preventDefault()
    if (!selectedFile) return
    const formElement = event.currentTarget
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const body = new FormData()
      body.append('file', selectedFile)
      const result = await callApi('/admin/documents', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body,
      })
      setDocuments((current) => [result.document, ...current])
      setSelectedFile(null)
      formElement.reset()
      setNotice('Document uploaded. The chatbot can now use its content as reference.')
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setBusy(false)
    }
  }

  async function deleteDocument(documentId) {
    setBusy(true)
    setError('')
    setNotice('')
    try {
      await callApi(`/admin/documents/${documentId}`, authOptions('DELETE'))
      setDocuments((current) => current.filter((document) => document.document_id !== documentId))
      setNotice('Document deleted.')
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setBusy(false)
    }
  }

  async function logout() {
    try {
      await callApi('/admin/logout', authOptions('POST'))
    } catch (requestError) {
      setError(requestError.message)
    }
    sessionStorage.removeItem('adminToken')
    setToken('')
    setDocuments([])
  }

  return (
    <div className="admin-shell">
      <header className="admin-topbar">
        <div className="brand-row"><div className="brand-mark"><ShieldCheck size={19} /></div><div><span className="brand-name">Civicline Admin</span><span className="brand-caption">KNOWLEDGE DASHBOARD</span></div></div>
        <div className="admin-top-actions">
          {token && <button className="button-muted" type="button" onClick={logout}>Sign out</button>}
        </div>
      </header>
      <main className="admin-content">
        <div className="section-kicker"><span className="live-dot" />ADMINISTRATION <span className="kicker-divider">/</span> KNOWLEDGE</div>
        <h1>Uploaded knowledge</h1>
        <p className="heading-subtitle">The chatbot answers from these documents and asks for the information needed to prepare a grievance.</p>
        {error && <div className="error-banner admin-error"><CircleHelp size={15} />{error}</div>}
        {notice && <p className="admin-notice">{notice}</p>}

        {!token ? <form className="admin-card admin-login" onSubmit={login}>
          <h2>Admin sign in</h2>
          <label className="admin-field"><span>Admin password</span><input type="password" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} required /></label>
          <button className="button-primary" type="submit" disabled={busy || !password}>{busy ? 'Signing in…' : 'Sign in'}</button>
          <p>Set <code>ADMIN_PASSWORD</code> in <code>backend/.env</code> before signing in.</p>
        </form> : <div className="document-grid">
          <form className="admin-card" onSubmit={uploadDocument}>
            <div className="review-head"><div><div className="review-eyebrow">KNOWLEDGE BASE</div><h2>Upload a topic document</h2></div></div>
            <p className="admin-empty">PDF and UTF-8 TXT files up to 10 MB. Scanned PDFs need OCR before upload. The chatbot will use matching content as its source and route unrelated questions to its customer-care response.</p>
            <label className="admin-field"><span>Choose a PDF or TXT file</span><input type="file" accept=".pdf,.txt,application/pdf,text/plain" onChange={(event) => setSelectedFile(event.target.files?.[0] || null)} required /></label>
            <button className="button-primary" type="submit" disabled={busy || !selectedFile}><FilePlus2 size={15} />{busy ? 'Uploading…' : 'Upload document'}</button>
          </form>

          <section className="admin-card">
            <div className="review-head"><div><div className="review-eyebrow">CHATBOT SOURCES</div><h2>Uploaded documents</h2></div></div>
            {documents.length === 0 ? <p className="admin-empty">No documents yet. Upload a reference document to enable topic-specific chatbot responses.</p> : <div className="document-list">
              {documents.map((document) => <article className="document-item" key={document.document_id}>
                <div><h3><FileText size={14} /> {document.filename}</h3><p>{document.character_count.toLocaleString()} extracted characters · Added {new Date(document.created_at).toLocaleDateString()}</p></div>
                <div className="document-actions"><button className="button-muted cancel-button" type="button" onClick={() => deleteDocument(document.document_id)} disabled={busy} aria-label={`Delete ${document.filename}`}><Trash2 size={14} />Delete</button></div>
              </article>)}
            </div>}
          </section>
        </div>}
      </main>
    </div>
  )
}
