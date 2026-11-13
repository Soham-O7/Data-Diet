import JobClient from './JobClient';

export function generateStaticParams() {
  return [{ jobId: 'latest' }, { jobId: 'demo' }];
}

export default function Page() {
  return <JobClient />;
}
