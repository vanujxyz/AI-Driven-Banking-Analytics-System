import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: './',
  css: { preprocessorOptions: { scss: { quietDeps: true, silenceDeprecations: ['import', 'global-builtin', 'mixed-decls'] } } },
});
